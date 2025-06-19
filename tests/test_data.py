import unittest

from unittest.mock import patch

from data import ImageFolder

import torch

from torchvision.transforms.v2 import Compose


class TestImageFolder(unittest.TestCase):
    @patch("data.walk")
    @patch("data.path.join")
    def setUp(self, mock_path_join, mock_walk):
        mock_walk.return_value = [
            ("/root", ("subdir",), ("image1.png", "image2.jpg", "not_image.txt"))
        ]

        mock_path_join.side_effect = lambda a, b: f"{a}/{b}"

        self.dataset = ImageFolder("/root", 4, 128)

    def test_initialization(self):
        self.assertEqual(len(self.dataset.image_paths), 2)

        self.assertIsInstance(self.dataset.input_transformer, Compose)
        self.assertIsInstance(self.dataset.target_transformer, Compose)

    def test_has_image_extension(self):
        self.assertTrue(self.dataset.has_image_extension("image.png"))
        self.assertFalse(self.dataset.has_image_extension("document.pdf"))

    @patch("data.decode_image")
    def test_getitem(self, mock_decode_image):
        mock_decode_image.return_value = torch.rand((3, 128, 128))

        x, y = self.dataset[0]

        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)

        self.assertEqual(x.shape, (3, 32, 32))
        self.assertEqual(y.shape, (3, 128, 128))

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)


if __name__ == "__main__":
    unittest.main()
