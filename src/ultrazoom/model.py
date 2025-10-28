from math import sqrt

from typing import Self

from functools import partial

import torch

from torch import Tensor

from torch.nn import (
    Module,
    ModuleList,
    Sequential,
    Linear,
    Conv2d,
    Sigmoid,
    SiLU,
    Upsample,
    PixelShuffle,
    AdaptiveAvgPool2d,
    Flatten,
    Parameter,
)

from torch.nn.utils.parametrize import (
    register_parametrization,
    is_parametrized,
    remove_parametrizations,
)

from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from huggingface_hub import PyTorchModelHubMixin


class UltraZoom(Module, PyTorchModelHubMixin):
    """
    A fast single-image super-resolution model with a deep low-resolution encoder network
    and high-resolution sub-pixel convolutional decoder head with global residual pathway.

    Ultra Zoom uses a "zoom in and enhance" approach to upscale images by first increasing
    the resolution of the input image using bicubic interpolation and then filling in the
    details using a deep neural network.
    """

    AVAILABLE_UPSCALE_RATIOS = {1, 2, 3, 4}

    AVAILABLE_HIDDEN_RATIOS = {1, 2, 4}

    def __init__(
        self,
        upscale_ratio: int,
        num_channels: int,
        hidden_ratio: int,
        num_encoder_layers: int,
    ):
        super().__init__()

        if upscale_ratio not in self.AVAILABLE_UPSCALE_RATIOS:
            raise ValueError(
                f"Upscale ratio must be either 2, 3, or 4, {upscale_ratio} given."
            )

        self.skip = Upsample(scale_factor=upscale_ratio, mode="bicubic")

        self.encoder = Encoder(num_channels, hidden_ratio, num_encoder_layers)

        self.decoder = SubpixelConv2d(num_channels, upscale_ratio)

        self.upscale_ratio = upscale_ratio

    @property
    def num_params(self) -> int:
        """Total number of parameters in the model."""

        return sum(param.numel() for param in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def freeze_parameters(self) -> None:
        """Freeze all model parameters to prevent them from being updated during training."""

        for param in self.parameters():
            param.requires_grad = False

    def add_weight_norms(self) -> None:
        """Add weight normalization parameterization to the network."""

        self.encoder.add_weight_norms()
        self.decoder.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Add LoRA adapters to all convolutional layers in the network."""

        self.encoder.add_lora_adapters(rank, alpha)
        self.decoder.add_lora_adapters(rank, alpha)

    def remove_parameterizations(self) -> None:
        """Remove all network parameterizations."""

        for module in self.modules():
            if is_parametrized(module):
                params = [name for name in module.parametrizations.keys()]

                for name in params:
                    remove_parametrizations(module, name)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Input image tensor of shape (B, C, H, W).
        """

        s = self.skip.forward(x)

        z = self.encoder.forward(x)
        z = self.decoder.forward(z)

        z = s + z  # Global residual connection

        return z, s

    @torch.inference_mode()
    def upscale(self, x: Tensor) -> Tensor:
        """
        Zoom and enhance the input image.

        Args:
            x: Input image tensor of shape (B, C, H, W).
        """

        z, _ = self.forward(x)

        z = torch.clamp(z, 0, 1)

        return z

    @torch.inference_mode()
    def test_compare(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Return both the zoomed and enhanced images for comparison.

        Args:
            x: Input image tensor of shape (B, C, H, W).
        """

        z, s = self.forward(x)

        z = torch.clamp(z, 0, 1)
        s = torch.clamp(s, 0, 1)

        return z, s


class Encoder(Module):
    """A low-resolution subnetwork employing a deep stack of encoder blocks."""

    def __init__(self, num_channels: int, hidden_ratio: int, num_layers: int):
        super().__init__()

        assert num_layers > 0, "Number of layers must be greater than 0."

        self.stem = Conv2d(3, num_channels, kernel_size=3, padding=1)

        self.body = ModuleList(
            [EncoderBlock(num_channels, hidden_ratio) for _ in range(num_layers)]
        )

        self.checkpoint = lambda layer, x: layer(x)

    def add_weight_norms(self) -> None:
        self.stem = weight_norm(self.stem)

        for layer in self.body:
            layer.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        register_parametrization(
            self.stem, "weight", ChannelLoRA(self.stem, rank, alpha)
        )

        for layer in self.body:
            layer.add_lora_adapters(rank, alpha)

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(self, x: Tensor) -> Tensor:
        z = self.stem.forward(x)

        for layer in self.body:
            z = self.checkpoint(layer, z)

        return z


class EncoderBlock(Module):
    """A single encoder block consisting of two stages and a residual connection."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        self.stage1 = SpatialAttention(num_channels)
        self.stage2 = InvertedBottleneck(num_channels, hidden_ratio)

    def add_weight_norms(self) -> None:
        self.stage1.add_weight_norms()
        self.stage2.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        self.stage1.add_lora_adapters(rank, alpha)
        self.stage2.add_lora_adapters(rank, alpha)

    def forward(self, x: Tensor) -> Tensor:
        z = self.stage1.forward(x)
        z = self.stage2.forward(z)

        z = x + z  # Local residual connection

        return z


class SpatialAttention(Module):
    """A spatial attention module with large depth-wise convolutions."""

    def __init__(self, num_channels: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."

        self.depthwise = Conv2d(
            num_channels,
            num_channels,
            kernel_size=11,
            padding=5,
            groups=num_channels,
            bias=False,
        )

        self.pointwise = Conv2d(num_channels, num_channels, kernel_size=1)

        self.sigmoid = Sigmoid()

    def add_weight_norms(self) -> None:
        self.depthwise = weight_norm(self.depthwise)
        self.pointwise = weight_norm(self.pointwise)

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

        z = self.sigmoid.forward(z)

        z = z * x

        return z


class InvertedBottleneck(Module):
    """A wide non-linear activation block with 3x3 convolutions."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_channels = hidden_ratio * num_channels

        self.conv1 = Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1)

        self.silu = SiLU()

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


class SubpixelConv2d(Module):
    """A high-resolution decoder using sub-pixel convolution."""

    def __init__(self, in_channels: int, upscale_ratio: int):
        super().__init__()

        assert upscale_ratio in {
            1,
            2,
            3,
            4,
        }, "Upscale ratio must be either 1, 2, 3, or 4."

        out_channels = 3 * upscale_ratio**2

        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.shuffle = PixelShuffle(upscale_ratio)

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


class ONNXModel(Module):
    """A wrapper class for exporting to ONNX format."""

    def __init__(self, model: UltraZoom):
        super().__init__()

        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model.upscale(x)


class Bouncer(Module):
    """A critic network for detecting real and fake images for adversarial training."""

    AVAILABLE_MODEL_SIZES = {"small", "medium", "large"}

    @classmethod
    def from_preconfigured(cls, model_size: str) -> Self:
        """Return a new preconfigured model."""

        assert model_size in cls.AVAILABLE_MODEL_SIZES, "Invalid model size."

        num_primary_layers = 3
        num_quaternary_layers = 3

        match model_size:
            case "small":
                num_primary_channels = 96
                num_secondary_channels = 192
                num_secondary_layers = 3
                num_tertiary_channels = 384
                num_tertiary_layers = 12
                num_quaternary_channels = 768

            case "medium":
                num_primary_channels = 128
                num_secondary_channels = 256
                num_secondary_layers = 6
                num_tertiary_channels = 512
                num_tertiary_layers = 24
                num_quaternary_channels = 1024

            case "large":
                num_primary_channels = 192
                num_secondary_channels = 384
                num_secondary_layers = 6
                num_tertiary_channels = 768
                num_tertiary_layers = 36
                num_quaternary_channels = 1536

        return cls(
            num_primary_channels,
            num_primary_layers,
            num_secondary_channels,
            num_secondary_layers,
            num_tertiary_channels,
            num_tertiary_layers,
            num_quaternary_channels,
            num_quaternary_layers,
        )

    def __init__(
        self,
        num_primary_channels: int,
        num_primary_layers: int,
        num_secondary_channels: int,
        num_secondary_layers: int,
        num_tertiary_channels: int,
        num_tertiary_layers: int,
        num_quaternary_channels: int,
        num_quaternary_layers: int,
    ):
        super().__init__()

        self.detector = Detector(
            num_primary_channels,
            num_primary_layers,
            num_secondary_channels,
            num_secondary_layers,
            num_tertiary_channels,
            num_tertiary_layers,
            num_quaternary_channels,
            num_quaternary_layers,
        )

        self.pool = AdaptiveAvgPool2d(1)

        self.flatten = Flatten(start_dim=1)

        self.classifier = BinaryClassifier(num_quaternary_channels)

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def add_spectral_norms(self) -> None:
        """Add spectral normalization to the network."""

        self.detector.add_spectral_norms()

    def remove_parameterizations(self) -> None:
        """Remove all parameterizations."""

        for module in self.modules():
            if is_parametrized(module):
                params = [name for name in module.parametrizations.keys()]

                for name in params:
                    remove_parametrizations(module, name)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z1, z2, z3, z4 = self.detector.forward(x)

        z5 = self.pool.forward(z4)
        z5 = self.flatten.forward(z5)

        z5 = self.classifier.forward(z5)

        return z1, z2, z3, z4, z5

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Return the probability that the input image is real."""

        _, _, _, _, z5 = self.forward(x)

        return z5


class Detector(Module):
    """A deep feature extraction network using convolutions."""

    def __init__(
        self,
        num_primary_channels: int,
        num_primary_layers: int,
        num_secondary_channels: int,
        num_secondary_layers: int,
        num_tertiary_channels: int,
        num_tertiary_layers: int,
        num_quaternary_channels: int,
        num_quaternary_layers: int,
    ):
        super().__init__()

        assert (
            num_primary_layers > 0
        ), "Number of primary layers must be greater than 0."

        assert (
            num_secondary_layers > 0
        ), "Number of secondary layers must be greater than 0."

        assert (
            num_tertiary_layers > 0
        ), "Number of tertiary layers must be greater than 0."

        assert (
            num_quaternary_layers > 0
        ), "Number of quaternary layers must be greater than 0."

        stage1 = Sequential(
            PixelCrush(3, num_primary_channels, 2),
            *[DetectorBlock(num_primary_channels) for _ in range(num_primary_layers)],
        )

        stage2 = Sequential(
            PixelCrush(num_primary_channels, num_secondary_channels, 2),
            *[
                DetectorBlock(num_secondary_channels)
                for _ in range(num_secondary_layers)
            ],
        )

        stage3 = Sequential(
            PixelCrush(num_secondary_channels, num_tertiary_channels, 2),
            *[DetectorBlock(num_tertiary_channels) for _ in range(num_tertiary_layers)],
        )

        stage4 = Sequential(
            PixelCrush(num_tertiary_channels, num_quaternary_channels, 2),
            *[
                DetectorBlock(num_quaternary_channels)
                for _ in range(num_quaternary_layers)
            ],
        )

        self.checkpoint = lambda layer, x: layer(x)

        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.stage4 = stage4

    def add_spectral_norms(self) -> None:
        for layer in self.stage1:
            layer.add_spectral_norms()

        for layer in self.stage2:
            layer.add_spectral_norms()

        for layer in self.stage3:
            layer.add_spectral_norms()

        for layer in self.stage4:
            layer.add_spectral_norms()

    def enable_activation_checkpointing(self) -> None:
        """
        Instead of memorizing the activations of the forward pass, recompute them
        at every encoder block.
        """

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        z1 = self.checkpoint(self.stage1.forward, x)
        z2 = self.checkpoint(self.stage2.forward, z1)
        z3 = self.checkpoint(self.stage3.forward, z2)
        z4 = self.checkpoint(self.stage4.forward, z3)

        return z1, z2, z3, z4


class PixelCrush(Module):
    """Downsample the feature maps using strided convolution."""

    def __init__(self, in_channels: int, out_channels: int, crush_factor: int):
        super().__init__()

        assert crush_factor in {
            1,
            2,
            3,
            4,
        }, "Crush factor must be either 1, 2, 3, or 4."

        self.conv = Conv2d(
            in_channels, out_channels, kernel_size=crush_factor, stride=crush_factor
        )

    def add_spectral_norms(self) -> None:
        self.conv = spectral_norm(self.conv)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class DetectorBlock(Module):
    """A detector block with depth-wise separable convolution and residual connection."""

    def __init__(self, num_channels: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."

        hidden_channels = 4 * num_channels

        self.conv1 = Conv2d(
            num_channels,
            num_channels,
            kernel_size=7,
            padding=3,
            groups=num_channels,
            bias=False,
        )

        self.conv2 = Conv2d(num_channels, hidden_channels, kernel_size=1)
        self.conv3 = Conv2d(hidden_channels, num_channels, kernel_size=1)

        self.silu = SiLU()

    def add_spectral_norms(self) -> None:
        self.conv1 = spectral_norm(self.conv1)
        self.conv2 = spectral_norm(self.conv2)
        self.conv3 = spectral_norm(self.conv3)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv1.forward(x)
        z = self.conv2.forward(z)
        z = self.silu.forward(z)
        z = self.conv3.forward(z)

        z = x + z  # Local residual connection

        return z


class BinaryClassifier(Module):
    """A simple two-layer binary classification head."""

    def __init__(self, input_features: int):
        super().__init__()

        assert input_features > 2, "Number of input features must be greater than 2."

        hidden_features = input_features // 2

        self.linear1 = Linear(input_features, hidden_features)
        self.linear2 = Linear(hidden_features, 1)

        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear1(x)
        z = self.silu(z)
        z = self.linear2(z)

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

    def forward(self, weight: Tensor) -> Tensor:
        z = self.lora_a @ self.lora_b

        z *= self.alpha

        # Move channels to front to match weight shape
        z = z.permute(2, 3, 0, 1)

        z = weight + z

        return z
