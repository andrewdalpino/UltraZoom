import torch

from torch import Tensor

from torch.nn import Module

from torchmetrics.classification import BinaryPrecision, BinaryRecall


class RelativisticF1Score(Module):
    """Computes the F1 score using relativistic mean predictions."""

    def __init__(self):
        super().__init__()

        self.precision_metric = BinaryPrecision()

        self.recall_metric = BinaryRecall()

    def update(
        self,
        y_pred_real: Tensor,
        y_pred_fake: Tensor,
        y_real: Tensor,
        y_fake: Tensor,
    ) -> None:
        y_pred_real -= y_pred_fake.mean()
        y_pred_fake -= y_pred_real.mean()

        y_pred = torch.cat((y_pred_real, y_pred_fake), dim=0)
        labels = torch.cat((y_real, y_fake), dim=0)

        self.precision_metric.update(y_pred, labels)
        self.recall_metric.update(y_pred, labels)

    def compute(self) -> tuple[Tensor, ...]:
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()

        if precision + recall == 0:
            f1_score = torch.tensor(0.0, device=precision.device)
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score, precision, recall

    def reset(self) -> None:
        self.precision_metric.reset()
        self.recall_metric.reset()
