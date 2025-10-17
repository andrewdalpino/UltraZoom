import unittest
import torch
from torch import Tensor
from src.ultrazoom.model import (
    UltraZoom,
    Encoder,
    EncoderBlock,
    SpatialAttention,
    InvertedBottleneck,
    SubpixelConv2d,
)


class BaseModelTest(unittest.TestCase):
    """Base class for model tests with common setup and utility methods."""

    def setUp(self):
        self.batch_size = 2
        self.channels = 3
        self.height = 64
        self.width = 64
        self.input_tensor = torch.rand(
            self.batch_size, self.channels, self.height, self.width
        )
        self.num_channels = 32
        self.hidden_ratio = 2

    def assert_output_shape(self, module, input_tensor, expected_shape):
        output = module(input_tensor)
        self.assertEqual(output.shape, expected_shape)

    def assert_forward_pass(self, module, input_tensor):
        try:
            output = module(input_tensor)
            self.assertIsInstance(output, Tensor)
            return output
        except Exception as e:
            self.fail(f"Forward pass failed with error: {e}")


class TestUltraZoom(BaseModelTest):
    """Tests for the UltraZoom class."""

    def setUp(self):
        super().setUp()
        self.upscale_ratio = 2
        self.num_encoder_layers = 3
        self.model = UltraZoom(
            upscale_ratio=self.upscale_ratio,
            num_channels=self.num_channels,
            hidden_ratio=self.hidden_ratio,
            num_encoder_layers=self.num_encoder_layers,
        )

    def test_initialization(self):
        self.assertEqual(self.model.upscale_ratio, self.upscale_ratio)
        self.assertIsInstance(self.model.encoder, Encoder)
        self.assertIsInstance(self.model.decoder, SubpixelConv2d)

    def test_invalid_upscale_ratio(self):
        with self.assertRaises(ValueError):
            UltraZoom(
                upscale_ratio=5,
                num_channels=self.num_channels,
                hidden_ratio=self.hidden_ratio,
                num_encoder_layers=self.num_encoder_layers,
            )

    def test_forward(self):
        output, skip = self.model(self.input_tensor)
        expected_shape = (
            self.batch_size,
            self.channels,
            self.height * self.upscale_ratio,
            self.width * self.upscale_ratio,
        )
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(skip.shape, expected_shape)

    def test_num_params(self):
        count = sum(param.numel() for param in self.model.parameters())
        self.assertEqual(self.model.num_params, count)

    def test_num_trainable_params(self):
        count = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        self.assertEqual(self.model.num_trainable_params, count)

    def test_weight_norms(self):
        initial_params = self.model.num_params
        self.model.add_weight_norms()

        # Check if any Conv2d layer has weight normalization
        has_norm = any(
            hasattr(module, "parametrizations")
            for module in self.model.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        self.assertTrue(has_norm)

        # Remove weight norms and check
        self.model.remove_weight_norms()
        has_norm = any(
            hasattr(module, "parametrizations") and bool(module.parametrizations)
            for module in self.model.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        self.assertFalse(has_norm)

    def test_upscale(self):
        with torch.no_grad():
            result = self.model.upscale(self.input_tensor)
        self.assertEqual(result.shape[:2], (self.batch_size, self.channels))
        self.assertEqual(
            result.shape[2:],
            (self.height * self.upscale_ratio, self.width * self.upscale_ratio),
        )
        self.assertEqual(result.dtype, torch.float32)

    def test_upscale_multi_step(self):
        steps = 2
        with torch.no_grad():
            result = self.model.upscale(self.input_tensor, steps=steps)
        expected_shape = (
            self.batch_size,
            self.channels,
            self.height * (self.upscale_ratio**steps),
            self.width * (self.upscale_ratio**steps),
        )
        self.assertEqual(result.shape, expected_shape)

    def test_test_compare(self):
        with torch.no_grad():
            result, skip = self.model.test_compare(self.input_tensor)
        self.assertEqual(result.shape, skip.shape)
        self.assertTrue(torch.all(result >= 0) and torch.all(result <= 1))
        self.assertTrue(torch.all(skip >= 0) and torch.all(skip <= 1))


class TestEncoder(BaseModelTest):
    """Tests for the Encoder class."""

    def setUp(self):
        super().setUp()
        self.num_layers = 4
        self.encoder = Encoder(self.num_channels, self.hidden_ratio, self.num_layers)

    def test_initialization(self):
        self.assertEqual(len(self.encoder.body), self.num_layers)
        self.assertIsInstance(self.encoder.stem, torch.nn.Conv2d)

    def test_forward(self):
        output = self.assert_forward_pass(self.encoder, self.input_tensor)
        self.assertEqual(
            output.shape, (self.batch_size, self.num_channels, self.height, self.width)
        )

    def test_activation_checkpointing(self):
        original_checkpoint = self.encoder.checkpoint
        self.encoder.enable_activation_checkpointing()
        self.assertNotEqual(original_checkpoint, self.encoder.checkpoint)

        # Make sure forward pass still works
        output = self.assert_forward_pass(self.encoder, self.input_tensor)
        self.assertEqual(
            output.shape, (self.batch_size, self.num_channels, self.height, self.width)
        )


class TestEncoderBlock(BaseModelTest):
    """Tests for the EncoderBlock class."""

    def setUp(self):
        super().setUp()
        self.encoder_block = EncoderBlock(self.num_channels, self.hidden_ratio)
        self.block_input = torch.rand(
            self.batch_size, self.num_channels, self.height, self.width
        )

    def test_initialization(self):
        self.assertIsInstance(self.encoder_block.stage1, SpatialAttention)
        self.assertIsInstance(self.encoder_block.stage2, InvertedBottleneck)

    def test_forward(self):
        output = self.assert_forward_pass(self.encoder_block, self.block_input)
        self.assertEqual(output.shape, self.block_input.shape)


class TestSpatialAttention(BaseModelTest):
    """Tests for the SpatialAttention class."""

    def setUp(self):
        super().setUp()
        self.spatial_attention = SpatialAttention(self.num_channels)
        self.attention_input = torch.rand(
            self.batch_size, self.num_channels, self.height, self.width
        )

    def test_initialization(self):
        self.assertIsInstance(self.spatial_attention.depthwise, torch.nn.Conv2d)
        self.assertIsInstance(self.spatial_attention.pointwise, torch.nn.Conv2d)
        self.assertIsInstance(self.spatial_attention.sigmoid, torch.nn.Sigmoid)

    def test_forward(self):
        output = self.assert_forward_pass(self.spatial_attention, self.attention_input)
        self.assertEqual(output.shape, self.attention_input.shape)


class TestInvertedBottleneck(BaseModelTest):
    """Tests for the InvertedBottleneck class."""

    def setUp(self):
        super().setUp()
        self.bottleneck = InvertedBottleneck(self.num_channels, self.hidden_ratio)
        self.bottleneck_input = torch.rand(
            self.batch_size, self.num_channels, self.height, self.width
        )

    def test_initialization(self):
        self.assertIsInstance(self.bottleneck.conv1, torch.nn.Conv2d)
        self.assertIsInstance(self.bottleneck.conv2, torch.nn.Conv2d)
        self.assertIsInstance(self.bottleneck.silu, torch.nn.SiLU)

        # Check hidden channels
        self.assertEqual(
            self.bottleneck.conv1.out_channels, self.num_channels * self.hidden_ratio
        )
        self.assertEqual(
            self.bottleneck.conv2.in_channels, self.num_channels * self.hidden_ratio
        )

    def test_forward(self):
        output = self.assert_forward_pass(self.bottleneck, self.bottleneck_input)
        self.assertEqual(output.shape, self.bottleneck_input.shape)

    def test_invalid_hidden_ratio(self):
        with self.assertRaises(AssertionError):
            InvertedBottleneck(self.num_channels, hidden_ratio=5)


class TestSubpixelConv2d(BaseModelTest):
    """Tests for the SubpixelConv2d class."""

    def setUp(self):
        super().setUp()
        self.upscale_ratio = 2
        self.subpixel = SubpixelConv2d(self.num_channels, self.upscale_ratio)
        self.subpixel_input = torch.rand(
            self.batch_size, self.num_channels, self.height, self.width
        )

    def test_initialization(self):
        self.assertIsInstance(self.subpixel.conv, torch.nn.Conv2d)
        self.assertIsInstance(self.subpixel.shuffle, torch.nn.PixelShuffle)

        # Check output channels calculation
        expected_out_channels = 3 * self.upscale_ratio**2
        self.assertEqual(self.subpixel.conv.out_channels, expected_out_channels)

    def test_forward(self):
        output = self.assert_forward_pass(self.subpixel, self.subpixel_input)
        expected_shape = (
            self.batch_size,
            3,
            self.height * self.upscale_ratio,
            self.width * self.upscale_ratio,
        )
        self.assertEqual(output.shape, expected_shape)

    def test_invalid_upscale_ratio(self):
        with self.assertRaises(AssertionError):
            SubpixelConv2d(self.num_channels, upscale_ratio=5)


if __name__ == "__main__":
    unittest.main()
