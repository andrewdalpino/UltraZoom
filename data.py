from os import path, walk
from warnings import warn

import torch

from torch import Tensor, linspace

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

from transforms import GaussianBlur, GaussianNoise, JPEGCompression


class ControlMix(Dataset):
    """
    A dataset of single HR images with a synthetic degradation function used to create the LR counterpart.
    """

    ALLOWED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})

    IMAGE_MODE = "RGB"

    def __init__(
        self,
        root_path: str,
        target_resolution: int,
        upscale_ratio: int,
        pre_transform: Transform | None,
        min_blur: float,
        max_blur: float,
        min_noise: float,
        max_noise: float,
        min_compression: float,
        max_compression: float,
    ):
        if target_resolution <= 0:
            raise ValueError(
                f"Target resolution must be positive, {target_resolution} given."
            )

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

        min_blur_sigma = min_blur * upscale_ratio
        max_blur_sigma = max_blur * upscale_ratio

        blur_transform = GaussianBlur(min_blur_sigma, max_blur_sigma)

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

        noise_transform = GaussianNoise(min_noise, max_noise)

        self.pre_transform = pre_transform
        self.blur_transform = blur_transform
        self.resize_transform = resize_transform
        self.noise_transform = noise_transform
        self.compression_transform = compression_transform
        self.to_tensor_transform = to_tensor_transform
        self.image_paths = image_paths

    @classmethod
    def has_image_extension(cls, filename: str) -> bool:
        _, extension = path.splitext(filename)

        return extension in cls.ALLOWED_EXTENSIONS

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[index]

        image = decode_image(image_path, mode=self.IMAGE_MODE)

        if self.pre_transform:
            image = self.pre_transform.forward(image)

        x, blur_sigma = self.blur_transform.forward(image)
        x = self.resize_transform.forward(x)
        x, compression = self.compression_transform.forward(x)
        x = self.to_tensor_transform.forward(x)
        x, noise_sigma = self.noise_transform.forward(x)

        c = torch.tensor([blur_sigma, noise_sigma, compression], dtype=torch.float32)

        y = self.to_tensor_transform.forward(image)

        return x, c, y

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
