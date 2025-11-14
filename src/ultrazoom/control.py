import torch

from torch import Tensor


class ControlVector:
    """A vector representing the strength of various enhancements."""

    def __init__(
        self,
        gaussian_blur: float,
        gaussian_noise: float,
        jpeg_compression: float,
    ):
        assert 0.0 <= gaussian_blur <= 1.0
        assert 0.0 <= gaussian_noise <= 1.0
        assert 0.0 <= jpeg_compression <= 1.0

        self.gaussian_blur = gaussian_blur
        self.gaussian_noise = gaussian_noise
        self.jpeg_compression = jpeg_compression

    def to_tensor(self) -> Tensor:
        return torch.tensor(
            [
                self.gaussian_blur,
                self.gaussian_noise,
                self.jpeg_compression,
            ]
        )
