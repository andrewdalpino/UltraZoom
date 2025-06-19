import torch

from torch import Tensor

from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    SiLU,
    Upsample,
    PixelShuffle,
)

from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from huggingface_hub import PyTorchModelHubMixin


class UltraZoom(Module, PyTorchModelHubMixin):
    """
    A fast single-image super-resolution model with a deep low-resolution encoder network
    and high-resolution sub-pixel convolutional decoder head with global residual pathway.
    """

    AVAILABLE_UPSCALE_RATIOS = {2, 4, 8}

    AVAILABLE_HIDDEN_RATIOS = {1, 2, 4}

    def __init__(
        self,
        upscale_ratio: int,
        num_channels: int,
        hidden_ratio: int,
        num_layers: int,
    ):
        super().__init__()

        if upscale_ratio not in self.AVAILABLE_UPSCALE_RATIOS:
            raise ValueError(
                f"Upscale ratio must be either 2, 4, or 8, {upscale_ratio} given."
            )

        if num_channels < 1:
            raise ValueError(
                f"Num channels must be greater than 0, {num_channels} given."
            )

        if hidden_ratio not in self.AVAILABLE_HIDDEN_RATIOS:
            raise ValueError(
                f"Hidden ratio must be either 1, 2, or 4, {hidden_ratio} given."
            )

        if num_layers < 1:
            raise ValueError(f"Num layers must be greater than 0, {num_layers} given.")

        self.input = weight_norm(Conv2d(3, num_channels, kernel_size=5, padding=2))

        self.skip = Upsample(scale_factor=upscale_ratio, mode="bicubic")

        self.encoder = Sequential(
            *[EncoderBlock(num_channels, hidden_ratio) for _ in range(num_layers)]
        )

        self.decoder = SubpixelConv2d(
            num_channels, upscale_ratio, kernel_size=3, padding=1
        )

        self.shuffle = PixelShuffle(upscale_ratio)

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def remove_weight_norms(self) -> None:
        for module in self.modules():
            if hasattr(module, "parametrizations"):
                params = [name for name in module.parametrizations.keys()]

                for name in params:
                    remove_parametrizations(module, name)

    def forward(self, x: Tensor) -> Tensor:
        z = self.input.forward(x)

        z = self.encoder.forward(z)
        z = self.decoder.forward(z)
        z = self.shuffle.forward(z)

        s = self.skip.forward(x)

        z += s  # Global residual connection

        return z

    @torch.no_grad()
    def upscale(self, x: Tensor) -> Tensor:
        z = self.forward(x)

        z = torch.clamp(z, 0, 1)

        return z


class EncoderBlock(Module):
    """A low-resolution encoder block with {num_channels} feature maps and wide activations."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        hidden_channels = hidden_ratio * num_channels

        conv1 = Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1)
        conv2 = Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1)

        self.conv1 = weight_norm(conv1)
        self.conv2 = weight_norm(conv2)

        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv1(x)
        z = self.silu(z)
        z = self.conv2(z)

        z += x  # Local residual connection

        return z


class SubpixelConv2d(Module):
    """A sub-pixel (1 / upscale_ratio) convolution layer with weight normalization."""

    def __init__(
        self, num_channels: int, upscale_ratio: int, kernel_size: int, padding: int
    ):
        super().__init__()

        channels_out = 3 * upscale_ratio**2

        conv = Conv2d(
            num_channels, channels_out, kernel_size=kernel_size, padding=padding
        )

        self.conv = weight_norm(conv)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
