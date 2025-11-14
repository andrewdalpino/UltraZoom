from os import path, walk
from warnings import warn

import torch

from torch import Tensor

from torch.utils.data import Dataset

from torchvision.io import decode_image

from torchvision.transforms.v2 import (
    Transform,
    RandomChoice,
    Resize,
    ToDtype,
)

from torchvision.transforms.v2.functional import InterpolationMode

from PIL import Image

from src.ultrazoom.control import ControlVector

from transforms import GaussianBlur, GaussianNoise, JPEGCompression


class ControlMix(Dataset):
    """
    A dataset of single HR images with a blind degradation function used to derive the LR counterpart.
    """

    ALLOWED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})

    IMAGE_MODE = "RGB"

    def __init__(
        self,
        root_path: str,
        target_resolution: int,
        upscale_ratio: int,
        pre_transform: Transform | None,
        min_gaussian_blur: float,
        max_gaussian_blur: float,
        min_gaussian_noise: float,
        max_gaussian_noise: float,
        min_compression: float,
        max_compression: float,
    ):
        if target_resolution <= 0:
            raise ValueError(
                f"Target resolution must be positive, {target_resolution} given."
            )

        if min_gaussian_blur == max_gaussian_blur:
            raise ValueError("Min and max Gaussian blur cannot be equal.")

        if min_gaussian_noise == max_gaussian_noise:
            raise ValueError("Min and max Gaussian noise cannot be equal.")

        if min_compression == max_compression:
            raise ValueError("Min and max compression cannot be equal.")

        image_paths = []
        dropped = 0

        for folder_path, _, filenames in walk(root_path):
            for filename in filenames:
                if self.has_image_extension(filename):
                    image_path = path.join(folder_path, filename)

                    image = Image.open(image_path)

                    width, height = image.size

                    if width < target_resolution or height < target_resolution:
                        dropped += 1

                        continue

                    image_paths.append(image_path)

        if dropped > 0:
            warn(
                f"Dropped {dropped} images that were smaller "
                f"than the target resolution of {target_resolution}."
            )

        blur_transform = GaussianBlur(min_gaussian_blur, max_gaussian_blur)

        gaussian_noise_transform = GaussianNoise(min_gaussian_noise, max_gaussian_noise)

        degraded_resolution = target_resolution // upscale_ratio

        resize_transform = RandomChoice(
            [
                Resize(
                    degraded_resolution,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                Resize(
                    degraded_resolution,
                    interpolation=InterpolationMode.BILINEAR,
                ),
            ]
        )

        compression_transform = JPEGCompression(min_compression, max_compression)

        to_tensor_transform = ToDtype(torch.float32, scale=True)

        to_image_transform = ToDtype(torch.uint8)

        self.pre_transform = pre_transform
        self.blur_transform = blur_transform
        self.gaussian_noise_transform = gaussian_noise_transform
        self.resize_transform = resize_transform
        self.compression_transform = compression_transform
        self.to_tensor_transform = to_tensor_transform
        self.to_image_transform = to_image_transform
        self.image_paths = image_paths
        self.min_gaussian_blur = min_gaussian_blur
        self.max_gaussian_blur = max_gaussian_blur
        self.min_gaussian_noise = min_gaussian_noise
        self.max_gaussian_noise = max_gaussian_noise
        self.min_compression = min_compression
        self.max_compression = max_compression

    @property
    def control_features(self) -> int:
        return 3

    @classmethod
    def has_image_extension(cls, filename: str) -> bool:
        _, extension = path.splitext(filename)

        return extension in cls.ALLOWED_EXTENSIONS

    def __getitem__(self, index: int) -> tuple[Tensor, ControlVector, Tensor]:
        image_path = self.image_paths[index]

        image = decode_image(image_path, mode=self.IMAGE_MODE)

        if self.pre_transform:
            image = self.pre_transform.forward(image)

        x, gaussian_blur_sigma = self.blur_transform.forward(image)
        x, gaussian_noise_sigma = self.gaussian_noise_transform.forward(x)
        x = self.resize_transform.forward(x)
        x, jpeg_compression = self.compression_transform.forward(x)
        x = self.to_tensor_transform.forward(x)

        gaussian_deblur = (gaussian_blur_sigma - self.min_gaussian_blur) / (
            self.max_gaussian_blur - self.min_gaussian_blur
        )

        gaussian_denoise = (gaussian_noise_sigma - self.min_gaussian_noise) / (
            self.max_gaussian_noise - self.min_gaussian_noise
        )

        jpeg_deartifact = (jpeg_compression - self.min_compression) / (
            self.max_compression - self.min_compression
        )

        c = ControlVector(
            gaussian_deblur,
            gaussian_denoise,
            jpeg_deartifact,
        ).to_tensor()

        y = self.to_tensor_transform.forward(image)

        return x, c, y

    def __len__(self):
        return len(self.image_paths)
