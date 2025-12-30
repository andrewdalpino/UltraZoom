import unittest
import torch
from torch import Tensor
from src.ultrazoom.model import (
    UltraZoom,
    ONNXModel,
    UNet,
    Encoder,
    EncoderBlock,
    ChannelControl,
    InvertedBottleneck,
    PixelCrush,
    Decoder,
    DecoderBlock,
    DecoderHead,
    SubpixelConv2d,
    ChannelLoRA,
    Bouncer,
    Detector,
    DetectorBlock,
    DepthwiseSeparableConv2d,
    BinaryClassifier,
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
        self.control_features = 3
        self.control_tensor = torch.rand(self.batch_size, self.control_features)

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
        self.model = UltraZoom(
            upscale_ratio=self.upscale_ratio,
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            control_features=self.control_features,
            hidden_ratio=self.hidden_ratio,
        )

    def test_initialization(self):
        self.assertEqual(self.model.upscale_ratio, self.upscale_ratio)
        self.assertEqual(self.model.control_features, self.control_features)
        self.assertIsInstance(self.model.stem, torch.nn.Conv2d)
        self.assertIsInstance(self.model.unet, UNet)
        self.assertIsInstance(self.model.head, DecoderHead)

    def test_invalid_upscale_ratio(self):
        with self.assertRaises(AssertionError):
            UltraZoom(
                upscale_ratio=5,
                primary_channels=32,
                primary_layers=2,
                secondary_channels=64,
                secondary_layers=2,
                tertiary_channels=128,
                tertiary_layers=2,
                quaternary_channels=256,
                quaternary_layers=2,
                control_features=self.control_features,
                hidden_ratio=self.hidden_ratio,
            )

    def test_forward(self):
        output = self.model(self.input_tensor, self.control_tensor)
        expected_shape = (
            self.batch_size,
            self.channels,
            self.height * self.upscale_ratio,
            self.width * self.upscale_ratio,
        )
        self.assertEqual(output.shape, expected_shape)

    def test_forward_with_mismatched_batch_size(self):
        bad_control = torch.rand(self.batch_size + 1, self.control_features)
        with self.assertRaises(AssertionError):
            self.model(self.input_tensor, bad_control)

    def test_forward_with_wrong_control_features(self):
        bad_control = torch.rand(self.batch_size, self.control_features + 1)
        with self.assertRaises(AssertionError):
            self.model(self.input_tensor, bad_control)

    def test_num_params(self):
        count = sum(param.numel() for param in self.model.parameters())
        self.assertEqual(self.model.num_params, count)

    def test_num_trainable_params(self):
        count = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        self.assertEqual(self.model.num_trainable_params, count)

    def test_freeze_parameters(self):
        self.model.freeze_parameters()
        for param in self.model.parameters():
            self.assertFalse(param.requires_grad)

    def test_weight_norms(self):
        self.model.add_weight_norms()
        has_norm = any(
            hasattr(module, "parametrizations")
            for module in self.model.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        self.assertTrue(has_norm)

        self.model.remove_parameterizations()
        has_norm = any(
            hasattr(module, "parametrizations") and bool(module.parametrizations)
            for module in self.model.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        self.assertFalse(has_norm)

    def test_lora_adapters(self):
        rank = 4
        alpha = 0.5
        self.model.add_lora_adapters(rank, alpha)
        has_lora = any(
            hasattr(module, "parametrizations")
            for module in self.model.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        self.assertTrue(has_lora)

    def test_activation_checkpointing(self):
        self.model.enable_activation_checkpointing()
        output = self.model(self.input_tensor, self.control_tensor)
        expected_shape = (
            self.batch_size,
            self.channels,
            self.height * self.upscale_ratio,
            self.width * self.upscale_ratio,
        )
        self.assertEqual(output.shape, expected_shape)

    def test_upscale(self):
        with torch.no_grad():
            result = self.model.upscale(self.input_tensor, self.control_tensor)
        self.assertEqual(result.shape[:2], (self.batch_size, self.channels))
        self.assertEqual(
            result.shape[2:],
            (self.height * self.upscale_ratio, self.width * self.upscale_ratio),
        )
        self.assertTrue(torch.all(result >= 0) and torch.all(result <= 1))


class TestONNXModel(BaseModelTest):
    """Tests for the ONNXModel wrapper."""

    def setUp(self):
        super().setUp()
        self.ultrazoom = UltraZoom(
            upscale_ratio=2,
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            control_features=self.control_features,
            hidden_ratio=self.hidden_ratio,
        )
        self.onnx_model = ONNXModel(self.ultrazoom)

    def test_forward(self):
        output = self.onnx_model(self.input_tensor, self.control_tensor)
        expected_shape = (
            self.batch_size,
            self.channels,
            self.height * 2,
            self.width * 2,
        )
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))


class TestUNet(BaseModelTest):
    """Tests for the UNet class."""

    def setUp(self):
        super().setUp()
        self.unet = UNet(
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            control_features=self.control_features,
            hidden_ratio=self.hidden_ratio,
        )

    def test_initialization(self):
        self.assertIsInstance(self.unet.encoder, Encoder)
        self.assertIsInstance(self.unet.decoder, Decoder)

    def test_invalid_layer_counts(self):
        with self.assertRaises(AssertionError):
            UNet(
                primary_channels=32,
                primary_layers=1,
                secondary_channels=64,
                secondary_layers=2,
                tertiary_channels=128,
                tertiary_layers=2,
                quaternary_channels=256,
                quaternary_layers=2,
                control_features=self.control_features,
                hidden_ratio=self.hidden_ratio,
            )

    def test_forward(self):
        input_features = torch.rand(self.batch_size, 32, self.height, self.width)
        output = self.unet(input_features, self.control_tensor)
        self.assertEqual(output.shape, (self.batch_size, 32, self.height, self.width))

    def test_activation_checkpointing(self):
        self.unet.enable_activation_checkpointing()
        input_features = torch.rand(self.batch_size, 32, self.height, self.width)
        output = self.unet(input_features, self.control_tensor)
        self.assertEqual(output.shape, (self.batch_size, 32, self.height, self.width))


class TestEncoder(BaseModelTest):
    """Tests for the Encoder class."""

    def setUp(self):
        super().setUp()
        self.encoder = Encoder(
            primary_channels=32,
            primary_layers=2,
            secondary_channels=64,
            secondary_layers=2,
            tertiary_channels=128,
            tertiary_layers=2,
            quaternary_channels=256,
            quaternary_layers=2,
            control_features=self.control_features,
            hidden_ratio=self.hidden_ratio,
        )

    def test_initialization(self):
        self.assertEqual(len(self.encoder.stage1), 2)
        self.assertEqual(len(self.encoder.stage2), 2)
        self.assertIsInstance(self.encoder.downsample1, PixelCrush)

    def test_invalid_layer_counts(self):
        with self.assertRaises(AssertionError):
            Encoder(
                primary_channels=32,
                primary_layers=0,
                secondary_channels=64,
                secondary_layers=2,
                tertiary_channels=128,
                tertiary_layers=2,
                quaternary_channels=256,
                quaternary_layers=2,
                control_features=self.control_features,
                hidden_ratio=self.hidden_ratio,
            )

    def test_forward(self):
        input_features = torch.rand(self.batch_size, 32, self.height, self.width)
        z1, z2, z3, z4 = self.encoder(input_features, self.control_tensor)
        self.assertEqual(z1.shape, (self.batch_size, 32, self.height, self.width))
        self.assertEqual(
            z2.shape, (self.batch_size, 64, self.height // 2, self.width // 2)
        )
        self.assertEqual(
            z3.shape, (self.batch_size, 128, self.height // 4, self.width // 4)
        )
        self.assertEqual(
            z4.shape, (self.batch_size, 256, self.height // 8, self.width // 8)
        )

    def test_activation_checkpointing(self):
        original_checkpoint = self.encoder.checkpoint
        self.encoder.enable_activation_checkpointing()
        self.assertNotEqual(original_checkpoint, self.encoder.checkpoint)

        input_features = torch.rand(self.batch_size, 32, self.height, self.width)
        z1, z2, z3, z4 = self.encoder(input_features, self.control_tensor)
        self.assertEqual(z1.shape[0], self.batch_size)


class TestEncoderBlock(BaseModelTest):
    """Tests for the EncoderBlock class."""

    def setUp(self):
        super().setUp()
        self.encoder_block = EncoderBlock(
            self.num_channels, self.control_features, self.hidden_ratio
        )
        self.block_input = torch.rand(
            self.batch_size, self.num_channels, self.height, self.width
        )

    def test_initialization(self):
        self.assertIsInstance(self.encoder_block.stage1, ChannelControl)
        self.assertIsInstance(self.encoder_block.stage2, InvertedBottleneck)

    def test_forward(self):
        output = self.encoder_block(self.block_input, self.control_tensor)
        self.assertEqual(output.shape, self.block_input.shape)

    def test_weight_norms(self):
        self.encoder_block.add_weight_norms()
        has_norm = any(
            hasattr(module, "parametrizations")
            for module in self.encoder_block.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        self.assertTrue(has_norm)

    def test_lora_adapters(self):
        self.encoder_block.add_lora_adapters(rank=4, alpha=0.5)
        has_lora = any(
            hasattr(module, "parametrizations")
            for module in self.encoder_block.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        self.assertTrue(has_lora)


class TestChannelControl(BaseModelTest):
    """Tests for the ChannelControl class."""

    def setUp(self):
        super().setUp()
        self.channel_control = ChannelControl(self.num_channels, self.control_features)
        self.control_input = torch.rand(
            self.batch_size, self.num_channels, self.height, self.width
        )

    def test_initialization(self):
        self.assertEqual(
            self.channel_control.scale.shape, (self.control_features, self.num_channels)
        )
        self.assertEqual(
            self.channel_control.shift.shape, (self.control_features, self.num_channels)
        )

    def test_invalid_parameters(self):
        with self.assertRaises(AssertionError):
            ChannelControl(0, self.control_features)
        with self.assertRaises(AssertionError):
            ChannelControl(self.num_channels, 0)

    def test_forward(self):
        output = self.channel_control(self.control_input, self.control_tensor)
        self.assertEqual(output.shape, self.control_input.shape)


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

        expected_hidden = self.num_channels * self.hidden_ratio
        self.assertEqual(self.bottleneck.conv1.out_channels, expected_hidden)
        self.assertEqual(self.bottleneck.conv2.in_channels, expected_hidden)

    def test_invalid_parameters(self):
        with self.assertRaises(AssertionError):
            InvertedBottleneck(0, self.hidden_ratio)
        with self.assertRaises(AssertionError):
            InvertedBottleneck(self.num_channels, hidden_ratio=5)

    def test_forward(self):
        output = self.bottleneck(self.bottleneck_input)
        self.assertEqual(output.shape, self.bottleneck_input.shape)

    def test_weight_norms(self):
        self.bottleneck.add_weight_norms()
        has_norm = any(
            hasattr(module, "parametrizations")
            for module in self.bottleneck.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        self.assertTrue(has_norm)

    def test_lora_adapters(self):
        self.bottleneck.add_lora_adapters(rank=4, alpha=0.5)
        has_lora = any(
            hasattr(module, "parametrizations")
            for module in self.bottleneck.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        self.assertTrue(has_lora)


class TestPixelCrush(BaseModelTest):
    """Tests for the PixelCrush class."""

    def setUp(self):
        super().setUp()
        self.crush_factor = 2
        self.pixel_crush = PixelCrush(self.num_channels, 64, self.crush_factor)
        self.crush_input = torch.rand(
            self.batch_size, self.num_channels, self.height, self.width
        )

    def test_initialization(self):
        self.assertIsInstance(self.pixel_crush.conv, torch.nn.Conv2d)

    def test_invalid_parameters(self):
        with self.assertRaises(AssertionError):
            PixelCrush(0, 64, 2)
        with self.assertRaises(AssertionError):
            PixelCrush(self.num_channels, 64, 5)

    def test_forward(self):
        output = self.pixel_crush(self.crush_input)
        expected_shape = (
            self.batch_size,
            64,
            self.height // self.crush_factor,
            self.width // self.crush_factor,
        )
        self.assertEqual(output.shape, expected_shape)

    def test_weight_norms(self):
        self.pixel_crush.add_weight_norms()
        self.assertTrue(hasattr(self.pixel_crush.conv, "parametrizations"))

    def test_spectral_norms(self):
        self.pixel_crush.add_spectral_norms()
        self.assertTrue(hasattr(self.pixel_crush.conv, "parametrizations"))


class TestDecoder(BaseModelTest):
    """Tests for the Decoder class."""

    def setUp(self):
        super().setUp()
        self.decoder = Decoder(
            primary_channels=256,
            primary_layers=2,
            secondary_channels=128,
            secondary_layers=2,
            tertiary_channels=64,
            tertiary_layers=2,
            quaternary_channels=32,
            quaternary_layers=2,
            control_features=self.control_features,
            hidden_ratio=self.hidden_ratio,
        )

    def test_initialization(self):
        self.assertEqual(len(self.decoder.stage1), 2)
        self.assertIsInstance(self.decoder.upsample1, SubpixelConv2d)

    def test_forward(self):
        z1 = torch.rand(self.batch_size, 256, self.height // 8, self.width // 8)
        z2 = torch.rand(self.batch_size, 128, self.height // 4, self.width // 4)
        z3 = torch.rand(self.batch_size, 64, self.height // 2, self.width // 2)
        z4 = torch.rand(self.batch_size, 32, self.height, self.width)

        output = self.decoder(z1, z2, z3, z4, self.control_tensor)
        self.assertEqual(output.shape, (self.batch_size, 32, self.height, self.width))

    def test_crop_feature_maps(self):
        x = torch.rand(self.batch_size, self.num_channels, 66, 66)
        cropped = self.decoder.crop_feature_maps(x, (64, 64))
        self.assertEqual(cropped.shape[2:], (64, 64))


class TestDecoderHead(BaseModelTest):
    """Tests for the DecoderHead class."""

    def setUp(self):
        super().setUp()
        self.upscale_ratio = 2
        self.head = DecoderHead(self.num_channels, self.upscale_ratio)
        self.head_input = torch.rand(
            self.batch_size, self.num_channels, self.height, self.width
        )

    def test_initialization(self):
        self.assertIsInstance(self.head.upsample, torch.nn.Sequential)

    def test_invalid_upscale_ratio(self):
        with self.assertRaises(AssertionError):
            DecoderHead(self.num_channels, 5)

    def test_forward(self):
        output = self.head(self.head_input)
        expected_shape = (
            self.batch_size,
            3,
            self.height * self.upscale_ratio,
            self.width * self.upscale_ratio,
        )
        self.assertEqual(output.shape, expected_shape)


class TestSubpixelConv2d(BaseModelTest):
    """Tests for the SubpixelConv2d class."""

    def setUp(self):
        super().setUp()
        self.upscale_ratio = 2
        self.out_channels = 3
        self.subpixel = SubpixelConv2d(
            self.num_channels, self.out_channels, self.upscale_ratio
        )
        self.subpixel_input = torch.rand(
            self.batch_size, self.num_channels, self.height, self.width
        )

    def test_initialization(self):
        self.assertIsInstance(self.subpixel.conv, torch.nn.Conv2d)
        self.assertIsInstance(self.subpixel.shuffle, torch.nn.PixelShuffle)

    def test_invalid_parameters(self):
        with self.assertRaises(AssertionError):
            SubpixelConv2d(0, self.out_channels, self.upscale_ratio)
        with self.assertRaises(AssertionError):
            SubpixelConv2d(self.num_channels, self.out_channels, 5)

    def test_forward(self):
        output = self.subpixel(self.subpixel_input)
        expected_shape = (
            self.batch_size,
            self.out_channels,
            self.height * self.upscale_ratio,
            self.width * self.upscale_ratio,
        )
        self.assertEqual(output.shape, expected_shape)

    def test_weight_norms(self):
        self.subpixel.add_weight_norms()
        self.assertTrue(hasattr(self.subpixel.conv, "parametrizations"))


class TestChannelLoRA(BaseModelTest):
    """Tests for the ChannelLoRA class."""

    def setUp(self):
        super().setUp()
        self.conv_layer = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.rank = 4
        self.alpha = 0.5
        self.lora = ChannelLoRA(self.conv_layer, self.rank, self.alpha)

    def test_initialization(self):
        out_channels, in_channels, h, w = self.conv_layer.weight.shape
        self.assertEqual(self.lora.lora_a.shape, (h, w, out_channels, self.rank))
        self.assertEqual(self.lora.lora_b.shape, (h, w, self.rank, in_channels))
        self.assertEqual(self.lora.alpha, self.alpha)

    def test_invalid_parameters(self):
        with self.assertRaises(AssertionError):
            ChannelLoRA(self.conv_layer, 0, self.alpha)
        with self.assertRaises(AssertionError):
            ChannelLoRA(self.conv_layer, self.rank, 0.0)

    def test_forward(self):
        weight = self.conv_layer.weight
        output = self.lora(weight)
        self.assertEqual(output.shape, weight.shape)


class TestBouncer(BaseModelTest):
    """Tests for the Bouncer discriminator."""

    def setUp(self):
        super().setUp()
        self.bouncer = Bouncer.from_preconfigured("small")

    def test_from_preconfigured(self):
        for size in ["small", "medium", "large"]:
            bouncer = Bouncer.from_preconfigured(size)
            self.assertIsInstance(bouncer, Bouncer)

    def test_invalid_model_size(self):
        with self.assertRaises(AssertionError):
            Bouncer.from_preconfigured("xlarge")

    def test_forward(self):
        z1, z2, z3, z4, prediction = self.bouncer(self.input_tensor)
        self.assertIsInstance(prediction, Tensor)
        self.assertEqual(prediction.shape, (self.batch_size, 1))

    def test_predict(self):
        with torch.no_grad():
            prediction = self.bouncer.predict(self.input_tensor)
        self.assertEqual(prediction.shape, (self.batch_size, 1))

    def test_spectral_norms(self):
        self.bouncer.add_spectral_norms()
        has_norm = any(
            hasattr(module, "parametrizations")
            for module in self.bouncer.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        self.assertTrue(has_norm)


class TestDetector(BaseModelTest):
    """Tests for the Detector class."""

    def setUp(self):
        super().setUp()
        self.detector = Detector(
            input_channels=3,
            primary_channels=64,
            primary_layers=2,
            secondary_channels=128,
            secondary_layers=2,
            tertiary_channels=256,
            tertiary_layers=2,
            quaternary_channels=512,
            quaternary_layers=2,
        )

    def test_initialization(self):
        self.assertIsInstance(self.detector.stage1, torch.nn.Sequential)
        self.assertIsInstance(self.detector.downsample1, PixelCrush)

    def test_invalid_input_channels(self):
        with self.assertRaises(AssertionError):
            Detector(
                input_channels=5,
                primary_channels=64,
                primary_layers=2,
                secondary_channels=128,
                secondary_layers=2,
                tertiary_channels=256,
                tertiary_layers=2,
                quaternary_channels=512,
                quaternary_layers=2,
            )

    def test_forward(self):
        z1, z2, z3, z4 = self.detector(self.input_tensor)
        self.assertEqual(z1.shape[1], 64)
        self.assertEqual(z2.shape[1], 128)
        self.assertEqual(z3.shape[1], 256)
        self.assertEqual(z4.shape[1], 512)


class TestDetectorBlock(BaseModelTest):
    """Tests for the DetectorBlock class."""

    def setUp(self):
        super().setUp()
        self.detector_block = DetectorBlock(self.num_channels, 4)
        self.block_input = torch.rand(
            self.batch_size, self.num_channels, self.height, self.width
        )

    def test_initialization(self):
        self.assertIsInstance(self.detector_block.conv1, DepthwiseSeparableConv2d)
        self.assertIsInstance(self.detector_block.conv2, torch.nn.Conv2d)
        self.assertIsInstance(self.detector_block.silu, torch.nn.SiLU)

    def test_forward(self):
        output = self.detector_block(self.block_input)
        self.assertEqual(output.shape, self.block_input.shape)

    def test_spectral_norms(self):
        self.detector_block.add_spectral_norms()
        has_norm = any(
            hasattr(module, "parametrizations")
            for module in self.detector_block.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        self.assertTrue(has_norm)


class TestDepthwiseSeparableConv2d(BaseModelTest):
    """Tests for the DepthwiseSeparableConv2d class."""

    def setUp(self):
        super().setUp()
        self.depthwise_sep = DepthwiseSeparableConv2d(
            self.num_channels, 64, kernel_size=7, padding=3
        )
        self.dsc_input = torch.rand(
            self.batch_size, self.num_channels, self.height, self.width
        )

    def test_initialization(self):
        self.assertIsInstance(self.depthwise_sep.depthwise, torch.nn.Conv2d)
        self.assertIsInstance(self.depthwise_sep.pointwise, torch.nn.Conv2d)
        self.assertEqual(self.depthwise_sep.depthwise.groups, self.num_channels)

    def test_forward(self):
        output = self.depthwise_sep(self.dsc_input)
        self.assertEqual(output.shape, (self.batch_size, 64, self.height, self.width))

    def test_spectral_norms(self):
        self.depthwise_sep.add_spectral_norms()
        self.assertTrue(hasattr(self.depthwise_sep.depthwise, "parametrizations"))
        self.assertTrue(hasattr(self.depthwise_sep.pointwise, "parametrizations"))


class TestBinaryClassifier(BaseModelTest):
    """Tests for the BinaryClassifier class."""

    def setUp(self):
        super().setUp()
        self.classifier = BinaryClassifier(128)
        self.classifier_input = torch.rand(self.batch_size, 128)

    def test_initialization(self):
        self.assertIsInstance(self.classifier.linear, torch.nn.Linear)
        self.assertEqual(self.classifier.linear.out_features, 1)

    def test_forward(self):
        output = self.classifier(self.classifier_input)
        self.assertEqual(output.shape, (self.batch_size, 1))


if __name__ == "__main__":
    unittest.main()
