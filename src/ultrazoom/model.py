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
    Identity,
)

from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
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
        """Add weight normalization parameterization to the network."""

        self.encoder.add_weight_norms()
        self.decoder.add_weight_norms()

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

    def add_weight_norms(self) -> None:
        self.conv = weight_norm(self.conv)

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

    def __init__(self, model_size: str):
        super().__init__()

        assert model_size in self.AVAILABLE_MODEL_SIZES, "Invalid model size."

        num_primary_layers = 3
        num_quaternary_layers = 3

        match model_size:
            case "small":
                num_primary_channels = 64
                num_secondary_channels = 128
                num_secondary_layers = 3
                num_tertiary_channels = 256
                num_tertiary_layers = 6
                num_quaternary_channels = 512

            case "medium":
                num_primary_channels = 96
                num_secondary_channels = 192
                num_secondary_layers = 3
                num_tertiary_channels = 384
                num_tertiary_layers = 12
                num_quaternary_channels = 768

            case "large":
                num_primary_channels = 128
                num_secondary_channels = 256
                num_secondary_layers = 6
                num_tertiary_channels = 512
                num_tertiary_layers = 24
                num_quaternary_channels = 1024

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

    def forward(self, x: Tensor) -> Tensor:
        z = self.detector.forward(x)

        z = self.pool.forward(z)
        z = self.flatten.forward(z)

        z = self.classifier.forward(z)

        return z


class Detector(Module):
    """A deep feature extractor network using convolutions."""

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

        self.stem = Conv2d(3, num_primary_channels, kernel_size=4, stride=4)

        stage1 = Sequential()

        stage1.extend(
            [
                DetectorBlock(num_primary_channels, num_primary_channels)
                for _ in range(num_primary_layers)
            ]
        )

        stage2 = Sequential()

        stage2.append(
            DetectorBlock(num_primary_channels, num_secondary_channels, stride=2)
        )

        stage2.extend(
            [
                DetectorBlock(num_secondary_channels, num_secondary_channels)
                for _ in range(num_secondary_layers - 1)
            ]
        )

        stage3 = Sequential()

        stage3.append(
            DetectorBlock(num_secondary_channels, num_tertiary_channels, stride=2)
        )

        stage3.extend(
            [
                DetectorBlock(num_tertiary_channels, num_tertiary_channels)
                for _ in range(num_tertiary_layers - 1)
            ]
        )

        stage4 = Sequential()

        stage4.append(
            DetectorBlock(num_tertiary_channels, num_quaternary_channels, stride=2)
        )

        stage4.extend(
            [
                DetectorBlock(num_quaternary_channels, num_quaternary_channels)
                for _ in range(num_quaternary_layers - 1)
            ]
        )

        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.stage4 = stage4

    def add_spectral_norms(self) -> None:
        self.stem = spectral_norm(self.stem)

        for layer in self.stage1:
            layer.add_spectral_norms()

        for layer in self.stage2:
            layer.add_spectral_norms()

        for layer in self.stage3:
            layer.add_spectral_norms()

        for layer in self.stage4:
            layer.add_spectral_norms()

    def forward(self, x: Tensor) -> Tensor:
        z = self.stem.forward(x)

        z = self.stage1.forward(z)
        z = self.stage2.forward(z)
        z = self.stage3.forward(z)
        z = self.stage4.forward(z)

        return z


class DetectorBlock(Module):
    """A detector block with depth-wise separable convolution and residual connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        assert in_channels > 0, "Number of channels must be greater than 0."
        assert out_channels > 0, "Number of output channels must be greater than 0."

        if in_channels == out_channels:
            skip = Sequential(Identity())
        else:
            skip = Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

        self.skip = skip

        hidden_channels = 4 * out_channels

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            padding=3,
            stride=stride,
            groups=in_channels,
            bias=False,
        )

        self.conv2 = Conv2d(out_channels, hidden_channels, kernel_size=1)
        self.conv3 = Conv2d(hidden_channels, out_channels, kernel_size=1)

        self.silu = SiLU()

    def add_spectral_norms(self) -> None:
        self.conv1 = spectral_norm(self.conv1)
        self.conv2 = spectral_norm(self.conv2)
        self.conv3 = spectral_norm(self.conv3)

    def forward(self, x: Tensor) -> Tensor:
        s = self.skip.forward(x)

        z = self.conv1.forward(x)
        z = self.conv2.forward(z)
        z = self.silu.forward(z)
        z = self.conv3.forward(z)

        z = s + z  # Local residual connection

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
