import random

from typing import Any

import torch

from torchvision import tv_tensors

from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2.functional import gaussian_blur, gaussian_noise, jpeg


class GaussianBlur(Transform):
    """
    A transform that applies Gaussian blur to an image.
    """

    def __init__(self, min_sigma: float, max_sigma: float):
        """
        Args:
            min_sigma (float): Minimum standard deviation of the Gaussian kernel.
            max_sigma (float): Maximum standard deviation of the Gaussian kernel.
        """

        super().__init__()

        assert min_sigma >= 0, f"Min sigma must be non-negative, {min_sigma} given."
        assert max_sigma >= 0, f"Max sigma must be non-negative, {max_sigma} given."

        assert max_sigma >= min_sigma, f"Max sigma must be greater than min sigma."

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sigma = random.uniform(self.min_sigma, self.max_sigma)

        return {"sigma": sigma}

    def transform(self, x: Any, params: dict[str, Any]) -> Any:
        sigma = params["sigma"]

        kernel_size = 2 * int(3 * sigma) + 1

        out = gaussian_blur(x, kernel_size, [sigma, sigma])

        return out, sigma


class PoissonNoise(Transform):
    """
    A transform that adds Poisson-distributed noise to an image.
    """

    def __init__(self, min_scale: float, max_scale: float):

        super().__init__()

        self.min_scale = min_scale
        self.max_scale = max_scale

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        scale = random.uniform(self.min_scale, self.max_scale)

        return {"scale": scale}

    def transform(self, x: Any, params: dict[str, Any]) -> Any:
        scale = params["scale"]

        out = torch.poisson(x * scale) / scale

        out = torch.clamp(out, 0.0, 1.0)

        out = tv_tensors.wrap(out, like=x)

        return out, scale


class GaussianNoise(Transform):
    """
    A transform that adds Gaussian-distributed noise to an image.
    """

    def __init__(self, min_sigma: float, max_sigma: float):
        """
        Args:
            min_sigma (float): Minimum standard deviation of the Gaussian noise to be added.
            max_sigma (float): Maximum standard deviation of the Gaussian noise to be added.
        """

        super().__init__()

        assert min_sigma >= 0, f"Min sigma must be non-negative, {min_sigma} given."
        assert max_sigma >= 0, f"Max sigma must be non-negative, {max_sigma} given."

        assert max_sigma >= min_sigma, f"Max sigma must be greater than min sigma."

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sigma = random.uniform(self.min_sigma, self.max_sigma)

        return {"sigma": sigma}

    def transform(self, x: Any, params: dict[str, Any]) -> Any:
        sigma = params["sigma"]

        out = gaussian_noise(x, mean=0, sigma=sigma, clip=True)

        return out, sigma


class JPEGCompression(Transform):
    """
    A transform that applies JPEG compression to an image.
    """

    def __init__(self, min_compression: int, max_compression: int):
        """
        Args:
            min_compression (int): Minimum quality of the JPEG compression.
            max_compression (int): Maximum quality of the JPEG compression.
        """

        super().__init__()

        assert (
            0 <= min_compression <= 1
        ), f"Min compression must be between 0 and 1, {min_compression} given."

        assert (
            0 <= max_compression <= 1
        ), f"Max compression must be between 0 and 1, {max_compression} given."

        assert (
            max_compression >= min_compression
        ), f"Max compression must be greater than min compression."

        self.min_compression = min_compression
        self.max_compression = max_compression

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        compression = random.uniform(self.min_compression, self.max_compression)

        return {"compression": compression}

    def transform(self, x: Any, params: dict[str, Any]) -> Any:
        compression = params["compression"]

        quality = int(100 * (1 - compression))

        out = jpeg(x, quality)

        return out, compression
