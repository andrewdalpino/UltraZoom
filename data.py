from os import path, walk
from warnings import warn

import torch

from torch import Tensor, linspace

from torch.utils.data import Dataset

from torchvision.io import decode_image

from torchvision.transforms.v2 import (
    Transform,
    Compose,
    RandomChoice,
    Resize,
    GaussianBlur,
    GaussianNoise,
    JPEG,
    ToDtype,
)

from torchvision.transforms.v2.functional import InterpolationMode

from PIL import Image

from src.ultrazoom.model import UltraZoom


class ImageFolder(Dataset):
    """
    A dataset of single HR images with a synthetic degradation function
    used to create the LR counterpart.
    """

    ALLOWED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})

    IMAGE_MODE = "RGB"

    def __init__(
        self,
        root_path: str,
        target_resolution: int,
        upscale_ratio: int,
        pre_transformer: Transform | None,
        blur_amount: float,
        min_noise: float,
        max_noise: float,
        min_compression: float,
        max_compression: float,
    ):
        if upscale_ratio not in UltraZoom.AVAILABLE_UPSCALE_RATIOS:
            raise ValueError(
                f"Upscale ratio must be either 2, 3, or 4, {upscale_ratio} given."
            )

        if target_resolution % upscale_ratio != 0:
            raise ValueError(
                f"Target resolution must divide evenly into upscale_ratio."
            )

        if blur_amount < 0.0:
            raise ValueError(f"Blur amount must be non-negative, {blur_amount} given.")

        if min_noise < 0.0:
            raise ValueError(
                f"Min noise amount must be non-negative, {min_noise} given."
            )

        if max_noise < min_noise:
            raise ValueError(f"Max noise amount must be greater than min noise.")

        if min_compression < 0.0 or min_compression > 1.0:
            raise ValueError(
                f"Min compression must be between 0 and 1, {min_compression} given."
            )

        if max_compression < min_compression:
            raise ValueError(f"Max compression must be greater than min compression.")

        blur_sigma = blur_amount * upscale_ratio
        blur_kernel_size = 2 * int(3 * blur_sigma) + 1

        degraded_resolution = target_resolution // upscale_ratio

        min_degraded_quality = 100 - int(max_compression * 100)
        max_degraded_quality = 100 - int(min_compression * 100)

        noise_amounts = linspace(min_noise, max_noise, steps=5).tolist()

        degrade_transformer = Compose(
            [
                GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma),
                RandomChoice(
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
                ),
                JPEG(quality=(min_degraded_quality, max_degraded_quality)),
                ToDtype(torch.float32, scale=True),
                RandomChoice(
                    [
                        GaussianNoise(sigma=noise_amount)
                        for noise_amount in noise_amounts
                    ]
                ),
            ]
        )

        target_transformer = ToDtype(torch.float32, scale=True)

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

        self.pre_transformer = pre_transformer
        self.degrade_transformer = degrade_transformer
        self.target_transformer = target_transformer
        self.image_paths = image_paths

    @classmethod
    def has_image_extension(cls, filename: str) -> bool:
        _, extension = path.splitext(filename)

        return extension in cls.ALLOWED_EXTENSIONS

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[index]

        image = decode_image(image_path, mode=self.IMAGE_MODE)

        if self.pre_transformer:
            image = self.pre_transformer(image)

        x = self.degrade_transformer(image)
        y = self.target_transformer(image)

        return x, y

    def __len__(self):
        return len(self.image_paths)


class ImagePairs(Dataset):
    """
    A dataset consisting of paired LR and HR images with the same name but in
    separate folders.
    """

    ALLOWED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})

    IMAGE_MODE = "RGB"

    def __init__(self, lr_root_path: str, hr_root_path: str):
        lr_image_paths = []
        hr_image_paths = []

        batch = [
            (lr_root_path, lr_image_paths),
            (hr_root_path, hr_image_paths),
        ]

        for root_path, image_paths in batch:
            for folder_path, _, filenames in walk(root_path):
                for filename in filenames:
                    if self.has_image_extension(filename):
                        image_path = path.join(folder_path, filename)

                        image_paths.append(image_path)

        self.lr_image_paths = lr_image_paths
        self.hr_image_paths = hr_image_paths

        self.transformer = ToDtype(torch.float32, scale=True)

    @classmethod
    def has_image_extension(cls, filename: str) -> bool:
        _, extension = path.splitext(filename)

        return extension in cls.ALLOWED_EXTENSIONS

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        lr_image_path = self.lr_image_paths[index]
        hr_image_path = self.hr_image_paths[index]

        lr_image = decode_image(lr_image_path, mode=self.IMAGE_MODE)
        hr_image = decode_image(hr_image_path, mode=self.IMAGE_MODE)

        x = self.transformer(lr_image)
        y = self.transformer(hr_image)

        return x, y

    def __len__(self):
        return len(self.lr_image_paths)
