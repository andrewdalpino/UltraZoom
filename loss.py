from torch import Tensor

from torch.nn import Module, MSELoss

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

        vgg22_loss = self.mse(y_pred_vgg22, y_vgg22)

        y_pred_vgg54 = self.vgg54.forward(y_pred_vgg22)
        y_vgg54 = self.vgg54.forward(y_vgg22)

        vgg54_loss = self.mse(y_pred_vgg54, y_vgg54)

        return vgg22_loss, vgg54_loss
