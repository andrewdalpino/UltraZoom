from math import sqrt, floor, ceil, log2

from typing import Self

from functools import partial

import torch

from torch import Tensor

from torch.nn import (
    Module,
    ModuleList,
    Sequential,
    Upsample,
    Conv2d,
    SiLU,
    Sigmoid,
    PixelShuffle,
    AdaptiveAvgPool2d,
    Flatten,
    Parameter,
)

from torch.nn.init import kaiming_uniform_
from torch.nn.functional import pad

from torch.nn.utils.parametrize import (
    register_parametrization,
    is_parametrized,
    remove_parametrizations,
)

from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from huggingface_hub import PyTorchModelHubMixin


type FeatureMapSize = tuple[int, int] | list[int]


class UltraZoom(Module, PyTorchModelHubMixin):
    """
    A model for image super-resolution based on a U-Net with adaptive residual
    connections.
    """

    AVAILABLE_UPSCALE_RATIOS = {2, 4, 8}

    def __init__(
        self,
        upscale_ratio: int,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
        hidden_ratio: int,
    ):
        super().__init__()

        assert (
            upscale_ratio in self.AVAILABLE_UPSCALE_RATIOS
        ), f"Upscale ratio must be one of {self.AVAILABLE_UPSCALE_RATIOS}, but got {upscale_ratio}."

        self.bicubic = Upsample(scale_factor=upscale_ratio, mode="bicubic")

        self.stem = FanOutProjection(3, primary_channels)

        self.unet = UNet(
            primary_channels,
            primary_layers,
            secondary_channels,
            secondary_layers,
            tertiary_channels,
            tertiary_layers,
            quaternary_channels,
            quaternary_layers,
            hidden_ratio,
        )

        self.head = SuperResolver(primary_channels, hidden_ratio, upscale_ratio)

        self.skip = ResidualConnection()

        self.upscale_ratio = upscale_ratio

    @property
    def num_params(self) -> int:
        """Total number of parameters in the model."""

        return sum(param.numel() for param in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def initialize_weights(self) -> None:
        """Initialize all model weights using Kaiming normal initialization."""

        self.stem.initialize_weights()
        self.unet.initialize_weights()
        self.head.initialize_weights()

    def freeze_parameters(self) -> None:
        """Freeze all model parameters to prevent them from being updated during training."""

        for param in self.parameters():
            param.requires_grad = False

    def add_weight_norms(self) -> None:
        """Add weight normalization parameterization to the network."""

        self.stem.add_weight_norms()
        self.unet.add_weight_norms()
        self.head.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Add LoRA adapters to all layers in the network."""

        self.stem.add_lora_adapters(rank, alpha)
        self.unet.add_lora_adapters(rank, alpha)
        self.head.add_lora_adapters(rank, alpha)

    def remove_parameterizations(self) -> None:
        """Remove all network parameterizations."""

        for module in self.modules():
            if is_parametrized(module):
                params = [name for name in module.parametrizations.keys()]

                for name in params:
                    remove_parametrizations(module, name)

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder and decoder block.
        """

        self.unet.enable_activation_checkpointing()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W).

        """

        s = self.bicubic.forward(x)

        z = self.stem.forward(x)
        z = self.unet.forward(z)
        z = self.head.forward(z)

        z = self.skip.forward(s, z)

        return z

    @torch.inference_mode()
    def upscale(self, x: Tensor) -> Tensor:
        """
        Convenience method for inference.

        Args:
            x: Input image tensor of shape (B, 3, H, W).
        """

        z = self.forward(x)

        z = torch.clamp(z, 0, 1)

        return z


class ONNXModel(Module):
    """A wrapper class for exporting to ONNX format."""

    def __init__(self, model: UltraZoom):
        super().__init__()

        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W).
        """

        return self.model.upscale(x)


class FanOutProjection(Module):
    """A linear projection for expanding the number of channels."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        assert in_channels > 0, "Input channels must be greater than 0."

        assert (
            in_channels < out_channels
        ), "Output channels must be greater than input channels."

        self.conv = Conv2d(in_channels, out_channels, kernel_size=1)

    def initialize_weights(self) -> None:
        kaiming_uniform_(self.conv.weight)

    def add_weight_norms(self) -> None:
        self.conv = weight_norm(self.conv)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        register_parametrization(
            self.conv,
            "weight",
            ChannelLoRA(self.conv, rank, alpha),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv.forward(x)

        return z


class UNet(Module):
    """
    An encoder/decoder network with adaptive residual connections.
    """

    def __init__(
        self,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
        hidden_ratio: int,
    ):
        super().__init__()

        assert primary_layers > 1, "Number of primary layers must be greater than 1."

        assert (
            secondary_layers > 1
        ), "Number of secondary layers must be greater than 1."

        assert tertiary_layers > 1, "Number of tertiary layers must be greater than 1."

        assert (
            quaternary_layers > 1
        ), "Number of quaternary layers must be greater than 1."

        self.encoder = Encoder(
            primary_channels,
            ceil(primary_layers / 2),
            secondary_channels,
            ceil(secondary_layers / 2),
            tertiary_channels,
            ceil(tertiary_layers / 2),
            quaternary_channels,
            ceil(quaternary_layers / 2),
            hidden_ratio,
        )

        self.decoder = Decoder(
            quaternary_channels,
            floor(quaternary_layers / 2),
            tertiary_channels,
            floor(tertiary_layers / 2),
            secondary_channels,
            floor(secondary_layers / 2),
            primary_channels,
            floor(primary_layers / 2),
            hidden_ratio,
        )

    def initialize_weights(self) -> None:
        self.encoder.initialize_weights()
        self.decoder.initialize_weights()

    def add_weight_norms(self) -> None:
        self.encoder.add_weight_norms()
        self.decoder.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        self.encoder.add_lora_adapters(rank, alpha)
        self.decoder.add_lora_adapters(rank, alpha)

    def enable_activation_checkpointing(self) -> None:
        self.encoder.enable_activation_checkpointing()
        self.decoder.enable_activation_checkpointing()

    def forward(self, x: Tensor) -> Tensor:
        z1, z2, z3, z4 = self.encoder.forward(x)

        z = self.decoder.forward(z4, z3, z2, z1)

        return z


class Encoder(Module):
    """An encoder subnetwork employing a deep stack of encoder blocks."""

    def __init__(
        self,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
        hidden_ratio: int,
    ):
        super().__init__()

        assert primary_layers > 0, "Number of primary layers must be greater than 0."

        assert (
            secondary_layers > 0
        ), "Number of secondary layers must be greater than 0."

        assert tertiary_layers > 0, "Number of tertiary layers must be greater than 0."

        assert (
            quaternary_layers > 0
        ), "Number of quaternary layers must be greater than 0."

        self.stage1 = ModuleList(
            [
                EncoderBlock(primary_channels, hidden_ratio)
                for _ in range(primary_layers)
            ]
        )

        self.stage2 = ModuleList(
            [
                EncoderBlock(secondary_channels, hidden_ratio)
                for _ in range(secondary_layers)
            ]
        )

        self.stage3 = ModuleList(
            [
                EncoderBlock(tertiary_channels, hidden_ratio)
                for _ in range(tertiary_layers)
            ]
        )

        self.stage4 = ModuleList(
            [
                EncoderBlock(quaternary_channels, hidden_ratio)
                for _ in range(quaternary_layers)
            ]
        )

        self.downsample1 = PixelCrush(primary_channels, secondary_channels, 2)
        self.downsample2 = PixelCrush(secondary_channels, tertiary_channels, 2)
        self.downsample3 = PixelCrush(tertiary_channels, quaternary_channels, 2)

        self.checkpoint = lambda layer, x: layer.forward(x)

    def initialize_weights(self) -> None:
        for layer in self.stage1:
            layer.initialize_weights()

        for layer in self.stage2:
            layer.initialize_weights()

        for layer in self.stage3:
            layer.initialize_weights()

        for layer in self.stage4:
            layer.initialize_weights()

        self.downsample1.initialize_weights()
        self.downsample2.initialize_weights()
        self.downsample3.initialize_weights()

    def add_weight_norms(self) -> None:
        for layer in self.stage1:
            layer.add_weight_norms()

        for layer in self.stage2:
            layer.add_weight_norms()

        for layer in self.stage3:
            layer.add_weight_norms()

        for layer in self.stage4:
            layer.add_weight_norms()

        self.downsample1.add_weight_norms()
        self.downsample2.add_weight_norms()
        self.downsample3.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        for layer in self.stage1:
            layer.add_lora_adapters(rank, alpha)

        for layer in self.stage2:
            layer.add_lora_adapters(rank, alpha)

        for layer in self.stage3:
            layer.add_lora_adapters(rank, alpha)

        for layer in self.stage4:
            layer.add_lora_adapters(rank, alpha)

        self.downsample1.add_lora_adapters(rank, alpha)
        self.downsample2.add_lora_adapters(rank, alpha)
        self.downsample3.add_lora_adapters(rank, alpha)

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z1 = x

        for layer in self.stage1:
            z1 = self.checkpoint(layer, z1)

        z2 = self.downsample1.forward(z1)

        for layer in self.stage2:
            z2 = self.checkpoint(layer, z2)

        z3 = self.downsample2.forward(z2)

        for layer in self.stage3:
            z3 = self.checkpoint(layer, z3)

        z4 = self.downsample3.forward(z3)

        for layer in self.stage4:
            z4 = self.checkpoint(layer, z4)

        return z1, z2, z3, z4


class EncoderBlock(Module):
    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        self.convnet = InvertedBottleneck(num_channels, hidden_ratio)

        self.skip = AdaptiveResidualMix(num_channels)

    def initialize_weights(self) -> None:
        self.convnet.initialize_weights()
        self.skip.initialize_weights()

    def add_weight_norms(self) -> None:
        self.convnet.add_weight_norms()
        self.skip.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        self.convnet.add_lora_adapters(rank, alpha)
        self.skip.add_lora_adapters(rank, alpha)

    def forward(self, x: Tensor) -> Tensor:
        z = self.convnet.forward(x)
        z = self.skip.forward(x, z)

        return z


class Decoder(Module):
    def __init__(
        self,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
        hidden_ratio: int,
    ):
        super().__init__()

        assert primary_layers > 0, "Number of primary layers must be greater than 0."

        assert (
            secondary_layers > 0
        ), "Number of secondary layers must be greater than 0."

        assert tertiary_layers > 0, "Number of tertiary layers must be greater than 0."

        assert (
            quaternary_layers > 0
        ), "Number of quaternary layers must be greater than 0."

        self.stage1 = ModuleList(
            [
                DecoderBlock(primary_channels, hidden_ratio)
                for _ in range(primary_layers)
            ]
        )

        self.stage2 = ModuleList(
            [
                DecoderBlock(secondary_channels, hidden_ratio)
                for _ in range(secondary_layers)
            ]
        )

        self.stage3 = ModuleList(
            [
                DecoderBlock(tertiary_channels, hidden_ratio)
                for _ in range(tertiary_layers)
            ]
        )

        self.stage4 = ModuleList(
            [
                DecoderBlock(quaternary_channels, hidden_ratio)
                for _ in range(quaternary_layers)
            ]
        )

        self.upsample1 = SubpixelConv2d(primary_channels, secondary_channels, 2)
        self.upsample2 = SubpixelConv2d(secondary_channels, tertiary_channels, 2)
        self.upsample3 = SubpixelConv2d(tertiary_channels, quaternary_channels, 2)

        self.skip1 = AdaptiveResidualMix(secondary_channels)
        self.skip2 = AdaptiveResidualMix(tertiary_channels)
        self.skip3 = AdaptiveResidualMix(quaternary_channels)

        self.checkpoint = lambda layer, x: layer.forward(x)

    def initialize_weights(self) -> None:
        for layer in self.stage1:
            layer.initialize_weights()

        for layer in self.stage2:
            layer.initialize_weights()

        for layer in self.stage3:
            layer.initialize_weights()

        for layer in self.stage4:
            layer.initialize_weights()

        self.upsample1.initialize_weights()
        self.upsample2.initialize_weights()
        self.upsample3.initialize_weights()

    def add_weight_norms(self) -> None:
        for layer in self.stage1:
            layer.add_weight_norms()

        for layer in self.stage2:
            layer.add_weight_norms()

        for layer in self.stage3:
            layer.add_weight_norms()

        for layer in self.stage4:
            layer.add_weight_norms()

        self.upsample1.add_weight_norms()
        self.upsample2.add_weight_norms()
        self.upsample3.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        for layer in self.stage1:
            layer.add_lora_adapters(rank, alpha)

        for layer in self.stage2:
            layer.add_lora_adapters(rank, alpha)

        for layer in self.stage3:
            layer.add_lora_adapters(rank, alpha)

        for layer in self.stage4:
            layer.add_lora_adapters(rank, alpha)

        self.upsample1.add_lora_adapters(rank, alpha)
        self.upsample2.add_lora_adapters(rank, alpha)
        self.upsample3.add_lora_adapters(rank, alpha)

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    @staticmethod
    def crop_feature_maps(x: Tensor, size: FeatureMapSize) -> Tensor:
        """
        Center-crop or pad the feature maps to the target size. This is sometimes
        necessary due to rounding during down/upsampling operations.
        """

        _, _, h, w = x.shape

        target_h, target_w = size

        if h > target_h:
            start_h = (h - target_h) // 2
            end_h = start_h + target_h

            x = x[:, :, start_h:end_h, :]

        elif h < target_h:
            pad_h = target_h - h

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            x = pad(x, (0, 0, pad_top, pad_bottom))

        if w > target_w:
            start_w = (w - target_w) // 2
            end_w = start_w + target_w

            x = x[:, :, :, start_w:end_w]

        elif w < target_w:
            pad_w = target_w - w

            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            x = pad(x, (pad_left, pad_right, 0, 0))

        return x

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor, x4: Tensor) -> Tensor:
        z = x1

        for layer in self.stage1:
            z = self.checkpoint(layer, z)

        z = self.upsample1.forward(z)

        z = self.crop_feature_maps(z, x2.shape[2:])

        z = self.skip1.forward(x2, z)

        for layer in self.stage2:
            z = self.checkpoint(layer, z)

        z = self.upsample2.forward(z)

        z = self.crop_feature_maps(z, x3.shape[2:])

        z = self.skip2.forward(x3, z)

        for layer in self.stage3:
            z = self.checkpoint(layer, z)

        z = self.upsample3.forward(z)

        z = self.crop_feature_maps(z, x4.shape[2:])

        z = self.skip3.forward(x4, z)

        for layer in self.stage4:
            z = self.checkpoint(layer, z)

        return z


class DecoderBlock(EncoderBlock):
    pass


class InvertedBottleneck(Module):
    """A wide non-linear activation block with convolutions."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_channels = hidden_ratio * num_channels

        self.conv1 = Conv2d(
            num_channels, hidden_channels, kernel_size=3, padding=1, bias=False
        )

        self.conv2 = Conv2d(
            hidden_channels, num_channels, kernel_size=3, padding=1, bias=False
        )

        self.silu = SiLU()

    def initialize_weights(self) -> None:
        kaiming_uniform_(self.conv1.weight)
        kaiming_uniform_(self.conv2.weight)

    def add_weight_norms(self) -> None:
        self.conv1 = weight_norm(self.conv1)
        self.conv2 = weight_norm(self.conv2)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        register_parametrization(
            self.conv1,
            "weight",
            ChannelLoRA(self.conv1, rank, alpha),
        )

        register_parametrization(
            self.conv2,
            "weight",
            ChannelLoRA(self.conv2, rank, alpha),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv1.forward(x)
        z = self.silu.forward(z)
        z = self.conv2.forward(z)

        return z


class ResidualConnection(Module):
    """
    An equally-weighted residual connection that adds the input to the residual feature maps.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        assert x.shape == z.shape, "Input and residual must have the same shape."

        return x + z


class AdaptiveResidualMix(Module):
    """
    A residual connection that adaptively mixes the input with the residual feature maps.
    """

    def __init__(self, num_channels: int):
        super().__init__()

        in_channels = 2 * num_channels

        self.conv = Conv2d(in_channels, num_channels, kernel_size=1, bias=False)

        self.alpha = Parameter(torch.Tensor([0.3]))

        self.sigmoid = Sigmoid()

    def initialize_weights(self) -> None:
        kaiming_uniform_(self.conv.weight)

    def add_weight_norms(self) -> None:
        self.conv = weight_norm(self.conv)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        register_parametrization(
            self.conv,
            "weight",
            ChannelLoRA(self.conv, rank, alpha),
        )

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        xz = torch.cat([x, z], dim=1)

        beta = self.conv.forward(xz)
        beta = self.sigmoid.forward(beta)

        # Alpha scalar allows learnable global identity mapping.
        alpha = self.sigmoid.forward(self.alpha)

        w = alpha * beta

        z_hat = (1 - w) * x + w * z

        return z_hat


class PixelCrush(Module):
    """Downsample the feature maps using strided convolution."""

    def __init__(self, in_channels: int, out_channels: int, crush_factor: int):
        super().__init__()

        assert in_channels > 0, "Input channels must be greater than 0."
        assert out_channels > 0, "Output channels must be greater than 0."

        assert crush_factor in {
            2,
            3,
            4,
        }, "Crush factor must be either 2, 3, or 4."

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=crush_factor,
            stride=crush_factor,
            bias=False,
        )

    def initialize_weights(self) -> None:
        kaiming_uniform_(self.conv.weight)

    def add_weight_norms(self) -> None:
        self.conv = weight_norm(self.conv)

    def add_spectral_norms(self) -> None:
        self.conv = spectral_norm(self.conv)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        register_parametrization(
            self.conv,
            "weight",
            ChannelLoRA(self.conv, rank, alpha),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class SubpixelConv2d(Module):
    """Upsample the feature maps using subpixel convolution."""

    def __init__(self, in_channels: int, out_channels: int, upscale_ratio: int):
        super().__init__()

        assert in_channels > 0, "Input channels must be greater than 0."
        assert out_channels > 0, "Output channels must be greater than 0."

        assert upscale_ratio in {
            2,
            3,
            4,
        }, "Upscale ratio must be either 2, 3, or 4."

        out_channels = out_channels * upscale_ratio**2

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,  # Effective stride will be 1 / upscale_ratio.
            padding=1,
            bias=False,
        )

        self.shuffle = PixelShuffle(upscale_ratio)

    def initialize_weights(self) -> None:
        kaiming_uniform_(self.conv.weight)

    def add_weight_norms(self) -> None:
        self.conv = weight_norm(self.conv)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        register_parametrization(
            self.conv,
            "weight",
            ChannelLoRA(self.conv, rank, alpha),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv.forward(x)
        z = self.shuffle.forward(z)

        return z


class SuperResolver(Module):
    """A decoder head for progressively upscaling the input feature maps beyond their original size."""

    def __init__(self, in_channels: int, hidden_ratio: int, upscale_ratio: int):
        super().__init__()

        assert upscale_ratio in {
            2,
            4,
            8,
        }, "Upscale ratio must be either 2, 4, or 8."

        num_layers = int(log2(upscale_ratio))

        self.layers = ModuleList(
            [
                SR2XBlock(in_channels, hidden_ratio, in_channels)
                for _ in range(num_layers - 1)
            ]
        )

        self.layers.append(SR2XBlock(in_channels, hidden_ratio, 3))

    def initialize_weights(self) -> None:
        for module in self.layers:
            module.initialize_weights()

    def add_weight_norms(self) -> None:
        for module in self.layers:
            module.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        for module in self.layers:
            module.add_lora_adapters(rank, alpha)

    def forward(self, z: Tensor) -> Tensor:
        for module in self.layers:
            z = module.forward(z)

        return z


class SR2XBlock(Module):
    """A 2X super-resolution block."""

    def __init__(self, in_channels: int, hidden_ratio: int, out_channels: int):
        super().__init__()

        self.refiner = DecoderBlock(in_channels, hidden_ratio)

        self.upscale = SubpixelConv2d(in_channels, out_channels, 2)

    def initialize_weights(self) -> None:
        self.refiner.initialize_weights()
        self.upscale.initialize_weights()

    def add_weight_norms(self) -> None:
        self.refiner.add_weight_norms()
        self.upscale.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        self.refiner.add_lora_adapters(rank, alpha)
        self.upscale.add_lora_adapters(rank, alpha)

    def forward(self, x: Tensor) -> Tensor:
        z = self.refiner.forward(x)
        z = self.upscale.forward(z)

        return z


class Bouncer(Module):
    """A critic network for detecting real and fake images for adversarial training."""

    AVAILABLE_MODEL_SIZES = {"small", "medium", "large"}

    @classmethod
    def from_preconfigured(cls, model_size: str) -> Self:
        """Return a new pre-configured model."""

        assert model_size in cls.AVAILABLE_MODEL_SIZES, "Invalid model size."

        primary_layers = 3
        quaternary_layers = 3

        match model_size:
            case "small":
                primary_channels = 64
                secondary_channels = 126
                secondary_layers = 4
                tertiary_channels = 256
                tertiary_layers = 6
                quaternary_channels = 512

            case "medium":
                primary_channels = 96
                secondary_channels = 192
                secondary_layers = 4
                tertiary_channels = 384
                tertiary_layers = 12
                quaternary_channels = 768

            case "large":
                primary_channels = 128
                secondary_channels = 256
                secondary_layers = 6
                tertiary_channels = 512
                tertiary_layers = 24
                quaternary_channels = 1024

        return cls(
            3,
            primary_channels,
            primary_layers,
            secondary_channels,
            secondary_layers,
            tertiary_channels,
            tertiary_layers,
            quaternary_channels,
            quaternary_layers,
        )

    def __init__(
        self,
        input_channels: int,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
    ):
        super().__init__()

        self.detector = FeatureDetector(
            input_channels,
            primary_channels,
            primary_layers,
            secondary_channels,
            secondary_layers,
            tertiary_channels,
            tertiary_layers,
            quaternary_channels,
            quaternary_layers,
        )

        self.head = FakeImageDiscriminator(quaternary_channels)

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def add_spectral_norms(self) -> None:
        """Add spectral normalization to the network."""

        self.detector.add_spectral_norms()
        self.head.add_spectral_norms()

    def remove_parameterizations(self) -> None:
        """Remove all parameterizations."""

        for module in self.modules():
            if is_parametrized(module):
                params = [name for name in module.parametrizations.keys()]

                for name in params:
                    remove_parametrizations(module, name)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z1, z2, z3, z4 = self.detector.forward(x)

        z5 = self.head.forward(z4)

        return z1, z2, z3, z4, z5

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Return the probability that the input image is real or fake."""

        _, _, _, _, z5 = self.forward(x)

        return z5


class FeatureDetector(Module):
    """A deep feature extraction network using convolutions."""

    def __init__(
        self,
        input_channels: int,
        primary_channels: int,
        primary_layers: int,
        secondary_channels: int,
        secondary_layers: int,
        tertiary_channels: int,
        tertiary_layers: int,
        quaternary_channels: int,
        quaternary_layers: int,
    ):
        super().__init__()

        assert input_channels in {1, 2, 3}, "Input channels must be either 1, 2, or 3."

        assert primary_layers > 0, "Number of primary layers must be greater than 0."

        assert (
            secondary_layers > 0
        ), "Number of secondary layers must be greater than 0."

        assert tertiary_layers > 0, "Number of tertiary layers must be greater than 0."

        assert (
            quaternary_layers > 0
        ), "Number of quaternary layers must be greater than 0."

        self.stage1 = Sequential(
            *[DetectorBlock(primary_channels, 4) for _ in range(primary_layers)],
        )

        self.stage2 = Sequential(
            *[DetectorBlock(secondary_channels, 4) for _ in range(secondary_layers)],
        )

        self.stage3 = Sequential(
            *[DetectorBlock(tertiary_channels, 4) for _ in range(tertiary_layers)],
        )

        self.stage4 = Sequential(
            *[DetectorBlock(quaternary_channels, 4) for _ in range(quaternary_layers)],
        )

        self.downsample1 = PixelCrush(input_channels, primary_channels, 2)
        self.downsample2 = PixelCrush(primary_channels, secondary_channels, 2)
        self.downsample3 = PixelCrush(secondary_channels, tertiary_channels, 2)
        self.downsample4 = PixelCrush(tertiary_channels, quaternary_channels, 2)

        self.checkpoint = lambda layer, x: layer(x)

    def add_spectral_norms(self) -> None:
        for layer in self.stage1:
            layer.add_spectral_norms()

        for layer in self.stage2:
            layer.add_spectral_norms()

        for layer in self.stage3:
            layer.add_spectral_norms()

        for layer in self.stage4:
            layer.add_spectral_norms()

        self.downsample1.add_spectral_norms()
        self.downsample2.add_spectral_norms()
        self.downsample3.add_spectral_norms()
        self.downsample4.add_spectral_norms()

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z1 = self.downsample1.forward(x)
        z1 = self.checkpoint(self.stage1.forward, z1)

        z2 = self.downsample2.forward(z1)
        z2 = self.checkpoint(self.stage2.forward, z2)

        z3 = self.downsample3.forward(z2)
        z3 = self.checkpoint(self.stage3.forward, z3)

        z4 = self.downsample4.forward(z3)
        z4 = self.checkpoint(self.stage4.forward, z4)

        return z1, z2, z3, z4


class DetectorBlock(Module):
    """A feature detector block with depth-wise separable convolution and adaptive residual connection."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_channels = hidden_ratio * num_channels

        self.conv1 = DepthwiseSeparableConv2d(
            num_channels, hidden_channels, kernel_size=7, padding=3
        )

        self.conv2 = Conv2d(hidden_channels, num_channels, kernel_size=1)

        self.skip = ResidualConnection()

        self.silu = SiLU()

    def add_spectral_norms(self) -> None:
        self.conv1.add_spectral_norms()

        self.conv2 = spectral_norm(self.conv2)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv1.forward(x)
        z = self.silu.forward(z)
        z = self.conv2.forward(z)

        z = self.skip.forward(x, z)

        return z


class DepthwiseSeparableConv2d(Module):
    """A depth-wise separable convolution layer."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int
    ):
        super().__init__()

        assert in_channels > 0, "Input channels must be greater than 0."
        assert out_channels > 0, "Output channels must be greater than 0."
        assert kernel_size > 0, "Kernel size must be greater than 0."
        assert padding >= 0, "Padding must be non-negative."

        self.depthwise = Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = Conv2d(in_channels, out_channels, kernel_size=1)

    def add_weight_norms(self) -> None:
        self.depthwise = weight_norm(self.depthwise)
        self.pointwise = weight_norm(self.pointwise)

    def add_spectral_norms(self) -> None:
        self.depthwise = spectral_norm(self.depthwise)
        self.pointwise = spectral_norm(self.pointwise)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        register_parametrization(
            self.depthwise,
            "weight",
            ChannelLoRA(self.depthwise, rank, alpha),
        )

        register_parametrization(
            self.pointwise,
            "weight",
            ChannelLoRA(self.pointwise, rank, alpha),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.depthwise.forward(x)
        z = self.pointwise.forward(z)

        return z


class FakeImageDiscriminator(Module):
    """
    A simple binary classification head that preserves positional invariance used
    to authenticate real images.
    """

    def __init__(self, num_channels: int):
        super().__init__()

        self.pool = AdaptiveAvgPool2d(1)

        self.conv = Conv2d(num_channels, 1, kernel_size=1)

        self.flatten = Flatten(start_dim=1)

    def add_spectral_norms(self) -> None:
        self.conv = spectral_norm(self.conv)

    def forward(self, x: Tensor) -> Tensor:
        z = self.pool.forward(x)
        z = self.conv.forward(z)

        z = self.flatten.forward(z)

        return z


class ChannelLoRA(Module):
    """Low rank channel decomposition transformation."""

    def __init__(self, layer: Conv2d, rank: int, alpha: float):
        super().__init__()

        assert rank > 0, "Rank must be greater than 0."
        assert alpha > 0.0, "Alpha must be greater than 0."

        out_channels, in_channels, h, w = layer.weight.shape

        lora_a = torch.randn(h, w, out_channels, rank) / sqrt(rank)
        lora_b = torch.zeros(h, w, rank, in_channels)

        self.lora_a = Parameter(lora_a)
        self.lora_b = Parameter(lora_b)

        self.alpha = alpha

    def forward(self, w: Tensor) -> Tensor:
        z = self.lora_a @ self.lora_b

        z *= self.alpha

        # Move channels to front to match weight shape.
        z = z.permute(2, 3, 0, 1)

        z = w + z

        return z
