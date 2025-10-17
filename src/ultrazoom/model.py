from functools import partial

import torch

from torch import Tensor

from torch.nn import (
    Module,
    ModuleList,
    Conv2d,
    Sigmoid,
    SiLU,
    Upsample,
    PixelShuffle,
)

from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
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

    AVAILABLE_UPSCALE_RATIOS = {1, 2, 3, 4, 8}

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

    def add_weight_norms(self) -> None:
        """Add weight normalization to all Conv2d layers in the model."""

        for module in self.modules():
            if isinstance(module, Conv2d):
                weight_norm(module)

    def remove_weight_norms(self) -> None:
        """Remove weight normalization parameterization."""

        for module in self.modules():
            if isinstance(module, Conv2d) and hasattr(module, "parametrizations"):
                params = [name for name in module.parametrizations.keys()]

                for name in params:
                    remove_parametrizations(module, name)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        s = self.skip.forward(x)

        z = self.encoder.forward(x)
        z = self.decoder.forward(z)

        z = s + z  # Global residual connection

        return z, s

    @torch.no_grad()
    def test_compare(self, x: Tensor) -> Tensor:
        z, s = self.forward(x)

        z = torch.clamp(z, 0, 1)
        s = torch.clamp(s, 0, 1)

        return z, s

    @torch.inference_mode()
    def upscale(self, x: Tensor, steps: int = 1) -> Tensor:
        """
        Upscale the input image by the specified number of steps.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).
            steps (int): Number of upscaling steps to perform. Default is 1.
        """

        assert steps > 0, "Number of steps must be greater than 0."

        for _ in range(steps):
            x, _ = self.forward(x)

            x = torch.clamp(x, 0, 1)

        return x


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
            8,
        }, "Upscale ratio must be either 1, 2, 3, 4, or 8."

        out_channels = 3 * upscale_ratio**2

        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.shuffle = PixelShuffle(upscale_ratio)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv.forward(x)
        z = self.shuffle.forward(z)

        return z
