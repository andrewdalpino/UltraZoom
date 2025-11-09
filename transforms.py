import torch

from torch import Tensor
from torch.nn import Module


class GreyNoise(Module):
    """
    A transform that adds Gaussian-distributed grey noise to an image.
    """

    def __init__(self, sigma: float = 0.1, clamp: bool = True):
        """
        Args:
            sigma (float): Standard deviation of the Gaussian noise to be added.
            clamp (bool): Whether to clamp the output to [0, 1].
        """

        super().__init__()

        self.sigma = sigma
        self.clamp = clamp

    def forward(self, image: Tensor) -> Tensor:
        c, h, w = image.shape

        noise = torch.randn((h, w)) * self.sigma

        noise = noise.unsqueeze(0).expand(c, -1, -1)

        out = image + noise

        if self.clamp:
            out = torch.clamp(out, 0, 1)

        return out
