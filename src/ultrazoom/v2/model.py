from functools import partial

import torch

from torch import Tensor

from torch.nn import (
    Module,
    ModuleList,
    Sequential,
    Conv2d,
    Upsample,
    SiLU,
)

from torch.nn.utils.parametrize import (
    register_parametrization,
    is_parametrized,
    remove_parametrizations,
)

from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from src.ultrazoom.model import (
    SpatialAttention,
    InvertedBottleneck,
    SubpixelConv2d,
    ChannelLoRA,
)

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

        self.bicubic = Upsample(scale_factor=upscale_ratio, mode="bicubic")

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

        s = self.bicubic.forward(x)

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
            [ResidualDenseBlock(num_channels, hidden_ratio) for _ in range(num_layers)]
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


class ResidualDenseBlock(Module):
    """A single encoder block consisting of three stages with dense connections."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        self.stage1 = Sequential(
            SpatialAttention(num_channels),
            InvertedBottleneck(num_channels, hidden_ratio),
        )

        self.stage2 = Sequential(
            SpatialAttention(2 * num_channels),
            InvertedBottleneck(2 * num_channels, hidden_ratio),
        )

        self.stage3 = Sequential(
            SpatialAttention(4 * num_channels),
            InvertedBottleneck(4 * num_channels, hidden_ratio),
        )

        self.mixer = Conv2d(8 * num_channels, num_channels, kernel_size=1, bias=False)

        self.silu = SiLU()

    def add_weight_norms(self) -> None:
        for stage in [self.stage1, self.stage2, self.stage3]:
            for layer in stage:
                layer.add_weight_norms()

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        for stage in [self.stage1, self.stage2, self.stage3]:
            for layer in stage:
                layer.add_lora_adapters(rank, alpha)

    def forward(self, x: Tensor) -> Tensor:
        z1 = self.stage1.forward(x)
        z2 = self.stage2.forward(torch.cat([x, z1], dim=1))
        z3 = self.stage3.forward(torch.cat([x, z1, z2], dim=1))

        z_hat = torch.cat([x, z1, z2, z3], dim=1)

        z = self.mixer.forward(z_hat)
        z = self.silu.forward(z)

        z = x + z

        return z


class ONNXModel(Module):
    """A wrapper class for exporting to ONNX format."""

    def __init__(self, model: UltraZoom):
        super().__init__()

        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model.upscale(x)
