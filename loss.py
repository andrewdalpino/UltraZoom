import torch

from torch import Tensor

from torch.nn import Module, MSELoss

from torchvision.models import vgg16, VGG16_Weights


class PerceptualL2Loss(Module):
    """Perceptual loss based on the L2 loss between VGG16 embeddings."""

    def __init__(self):
        super().__init__()

        model = vgg16(weights=VGG16_Weights).features

        for param in model.parameters():
            param.requires_grad = False

        model.eval()

        self.model = model
        self.mse = MSELoss()

    @property
    def num_params(self) -> int:
        return sum(param.numel() for param in self.model.parameters())

    def forward(self, y_pred: Tensor, y: Tensor) -> Tensor:
        input_embeddings = self.model.forward(y_pred)
        target_embeddings = self.model.forward(y)

        loss = self.mse(input_embeddings, target_embeddings)

        return loss


class TVLoss(Module):
    """Total variation (TV) penalty as a loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor) -> Tensor:
        b, c, h, w = y_pred.size()

        h_delta = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        w_delta = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

        h_variance = torch.pow(h_delta, 2).sum()
        w_variance = torch.pow(w_delta, 2).sum()

        h_variance /= b * c * (h - 1) * w
        w_variance /= b * c * h * (w - 1)

        penalty = w_variance + h_variance

        return penalty
