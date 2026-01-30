import torch

from torch import Tensor

from torch.nn import Module, MSELoss, BCEWithLogitsLoss, Parameter

from torchvision.models import vgg19, VGG19_Weights


class VGGLoss(Module):
    """
    A perceptual loss based on the L2 distance between low and high-level VGG19
    embeddings of the predicted and target image.
    """

    def __init__(self):
        super().__init__()

        model = vgg19(weights=VGG19_Weights.DEFAULT)

        for param in model.parameters():
            param.requires_grad = False

        model.eval()

        self.vgg22 = model.features[0:9]
        self.vgg54 = model.features[9:36]

        self.mse = MSELoss()

    @property
    def num_params(self) -> int:
        num_params = 0

        for module in (self.vgg22, self.vgg54):
            num_params += sum(param.numel() for param in module.parameters())

        return num_params

    def forward(self, y_pred: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        y_pred_vgg22 = self.vgg22.forward(y_pred)
        y_vgg22 = self.vgg22.forward(y)

        vgg22_loss = self.mse.forward(y_pred_vgg22, y_vgg22)

        y_pred_vgg54 = self.vgg54.forward(y_pred_vgg22)
        y_vgg54 = self.vgg54.forward(y_vgg22)

        vgg54_loss = self.mse.forward(y_pred_vgg54, y_vgg54)

        return vgg22_loss, vgg54_loss


class RelativisticBCELoss(Module):
    """
    Relativistic average BCE with logits loss for generative adversarial network training.
    """

    def __init__(self):
        super().__init__()

        self.bce = BCEWithLogitsLoss()

    def forward(
        self,
        y_pred_real: Tensor,
        y_pred_fake: Tensor,
        y_real: Tensor,
        y_fake: Tensor,
    ) -> Tensor:
        y_pred_real_hat = y_pred_real - y_pred_fake.mean()
        y_pred_fake_hat = y_pred_fake - y_pred_real.mean()

        y_pred = torch.cat((y_pred_real_hat, y_pred_fake_hat))
        y = torch.cat((y_real, y_fake))

        loss = self.bce.forward(y_pred, y)

        return loss


class BalancedMultitaskLoss(Module):
    """A dynamic multitask loss weighting where each task contributes equally."""

    def __init__(self):
        super().__init__()

    def forward(self, losses: Tensor) -> Tensor:
        combined_loss = losses / losses.detach()

        combined_loss = combined_loss.sum()

        return combined_loss


class AdaptiveMultitaskLoss(Module):
    """
    Adaptive loss weighting using homoscedastic i.e. task-dependent uncertainty as a training signal.
    """

    def __init__(self, num_losses: int):
        super().__init__()

        assert num_losses > 0, "Number of losses must be positive"

        self.log_sigmas = Parameter(torch.zeros(num_losses))

        self.num_losses = num_losses

    @property
    def loss_weights(self) -> Tensor:
        """
        Get current loss weights based on learned uncertainties.

        Returns:
            Tensor of loss weights for each task.
        """

        weights = torch.exp(-2.0 * self.log_sigmas)

        return weights

    def forward(self, losses: Tensor) -> Tensor:
        """
        Compute task uncertainty-weighted combined loss.

        Args:
            losses: Tensor of individual loss values for each task.

        Returns:
            Combined task uncertainty-weighted loss.
        """

        assert (
            losses.size(0) == self.num_losses
        ), "Number of losses must match number of tasks."

        weighted_losses = 0.5 * self.loss_weights * losses

        # Regularization term to prevent task weight collapse.
        weighted_losses += self.log_sigmas

        combined_loss = weighted_losses.sum()

        return combined_loss
