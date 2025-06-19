from unittest import main, TestCase

import torch

from torch import Tensor

from loss import TVLoss


class TestTVLoss(TestCase):
    def setUp(self):
        self.loss_function = TVLoss()

    def test_tv_loss_initialization(self):
        self.assertIsInstance(self.loss_function, TVLoss)

    def test_tv_loss_forward(self):
        y_pred = torch.rand((1, 3, 4, 4))

        penalty = self.loss_function(y_pred)

        self.assertIsInstance(penalty, Tensor)
        self.assertGreaterEqual(penalty.item(), 0)

    def test_tv_loss_forward_zero(self):
        y_pred = torch.zeros((1, 3, 4, 4))

        penalty = self.loss_function(y_pred)

        self.assertEqual(penalty.item(), 0)


if __name__ == "__main__":
    main()
