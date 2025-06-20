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

    Ultra Zoom uses a "zoom in and enhance" approach to upscale images by first increasing
    the resolution of the input image using bicubic interpolation and then filling in the
    details using a deep neural network.
    """

    AVAILABLE_UPSCALE_RATIOS = {2, 4, 8}

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

        if num_encoder_layers < 1:
            raise ValueError(
                f"Num layers must be greater than 0, {num_encoder_layers} given."
            )

        self.skip = Upsample(scale_factor=upscale_ratio, mode="bicubic")

        self.encoder = Encoder(num_channels, hidden_ratio, num_encoder_layers)
        self.decoder = Decoder(num_channels, upscale_ratio)

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

    def forward(self, x: Tensor) -> Tensor:
        s = self.skip.forward(x)

        z = self.encoder.forward(x)
        z = self.decoder.forward(z)

        s += z  # Global residual connection

        return s

    @torch.no_grad()
    def upscale(self, x: Tensor) -> Tensor:
        z = self.forward(x)

        z = torch.clamp(z, 0, 1)

        return z


class Encoder(Module):
    def __init__(self, num_channels: int, hidden_ratio: int, num_layers: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."
        assert num_layers > 0, "Number of layers must be greater than 0."

        self.input = Conv2d(3, num_channels, kernel_size=1)

        self.body = Sequential(
            *[EncoderBlock(num_channels, hidden_ratio) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.input.forward(x)

        z = self.body.forward(z)

        return z


class EncoderBlock(Module):
    """A low-resolution encoder block with {num_channels} feature maps and wide activations."""

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_channels = hidden_ratio * num_channels

        self.conv1 = Conv2d(num_channels, num_channels, kernel_size=7, padding=3)
        self.conv2 = Conv2d(num_channels, hidden_channels, kernel_size=1)
        self.conv3 = Conv2d(hidden_channels, num_channels, kernel_size=1)

        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        s = x.clone()

        z = self.conv1.forward(x)
        z = self.conv2.forward(z)
        z = self.silu.forward(z)
        z = self.conv3.forward(z)

        s += z  # Local residual connection

        return s


class Decoder(Module):
    """A high-resolution decoder head with sub-pixel convolution and pixel shuffling."""

    def __init__(self, num_channels: int, upscale_ratio: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."
        assert upscale_ratio in {2, 4, 8}, "Upscale ratio must be either 2, 4, or 8."

        channels_out = 3 * upscale_ratio**2

        self.subpixel_conv = Conv2d(
            num_channels, channels_out, kernel_size=7, padding=3
        )

        self.shuffle = PixelShuffle(upscale_ratio)

    def forward(self, x: Tensor) -> Tensor:
        z = self.subpixel_conv.forward(x)

        z = self.shuffle.forward(z)

        return z
