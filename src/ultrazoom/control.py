import torch

from torch import Tensor


class ControlVector:
    """A vector representing the strength of various enhancements."""

    def __init__(
        self,
        gaussian_deblur: float,
        gaussian_denoise: float,
        jpeg_deartifact: float,
    ):
        assert 0.0 <= gaussian_deblur <= 1.0
        assert 0.0 <= gaussian_denoise <= 1.0
        assert 0.0 <= jpeg_deartifact <= 1.0

        self.gaussian_deblur = gaussian_deblur
        self.gaussian_denoise = gaussian_denoise
        self.jpeg_deartifact = jpeg_deartifact

    def to_tensor(self) -> Tensor:
        return torch.tensor(
            [
                self.gaussian_deblur,
                self.gaussian_denoise,
                self.jpeg_deartifact,
            ]
        )
