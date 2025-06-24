from functools import partial

import torch

from torch import Tensor

from torch.nn import (
    Module,
    ModuleList,
    Upsample,
    Conv2d,
    Sigmoid,
    SiLU,
    PixelShuffle,
)

from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from huggingface_hub import PyTorchModelHubMixin


class UltraZoom(Module, PyTorchModelHubMixin):
    """
    A fast single-image super-resolution model. Ultra Zoom uses a "zoom in and enhance"
    approach to upscale images by first increasing the resolution of the input image
    using bicubic interpolation and then filling in the details using a deep neural
    network.
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

        self.upscale_ratio = upscale_ratio

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

        z = s + z  # Global residual connection

        return z

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

        self.stem = Conv2d(3, num_channels, kernel_size=11, padding=5)

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
    """
    A low-resolution encoder block using large depth-wise separable convolutions
    with element-wise attention and inverted bottleneck layers.
    """

    def __init__(self, num_channels: int, hidden_ratio: int):
        super().__init__()

        self.stage1 = DepthwiseSeparableConv2dWithAttention(num_channels)
        self.stage2 = InvertedBottleneck(num_channels, hidden_ratio)

    def forward(self, x: Tensor) -> Tensor:
        z = self.stage1.forward(x)
        z = self.stage2.forward(z)

        z = x + z  # Local residual connection

        return z


class DepthwiseSeparableConv2dWithAttention(Module):
    """
    A depth-wise separable convolution layer with a large field-of-view and
    element-wise attention.
    """

    def __init__(self, num_channels: int):
        super().__init__()

        assert num_channels > 0, "Number of channels must be greater than 0."

        self.depthwise = Conv2d(
            num_channels, num_channels, kernel_size=7, padding=3, groups=num_channels
        )

        self.attention = ElementwiseAttention(num_channels)

        self.pointwise = Conv2d(num_channels, num_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        z = self.depthwise.forward(x)
        z = self.attention.forward(z)
        z = self.pointwise.forward(z)

        return z


class ElementwiseAttention(Module):
    def __init__(self, num_channels: int):
        super().__init__()

        self.conv = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv.forward(x)
        z = self.sigmoid.forward(z)

        return z * x


class InvertedBottleneck(Module):
    """An inverted bottleneck layer with SiLU activation."""

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


class Decoder(Module):
    """A high-resolution decoder head with sub-pixel convolution."""

    def __init__(self, num_channels: int, upscale_ratio: int):
        super().__init__()

        self.conv = SubPixelConv2d(
            num_channels,
            num_channels,
            upscale_ratio=upscale_ratio,
            kernel_size=3,
            padding=1,
        )

        self.shuffle = PixelShuffle(upscale_ratio)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv.forward(x)
        z = self.shuffle.forward(z)

        return z


class SubPixelConv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_ratio: int,
        kernel_size: int,
        padding: int,
    ):
        super().__init__()

        assert upscale_ratio in {2, 4, 8}, "Upscale ratio must be either 2, 4, or 8."

        out_channels = 3 * upscale_ratio**2

        self.conv = Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv.forward(x)
