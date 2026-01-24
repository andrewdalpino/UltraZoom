import random

from argparse import ArgumentParser
from functools import partial

import torch

from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, SGD
from torch.amp import autocast
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.backends.mps import is_available as mps_is_available
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms.v2 import (
    Compose,
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
    ColorJitter,
)

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    VisualInformationFidelity,
)

from data import ImageFolder
from src.ultrazoom.model import UltraZoom
from loss import VGGLoss, AdaptiveMultitaskLoss

from tqdm import tqdm


def main():
    parser = ArgumentParser(description="Training script")

    parser.add_argument("--train_images_path", default="./dataset/train", type=str)
    parser.add_argument("--test_images_path", default="./dataset/test", type=str)
    parser.add_argument("--num_dataset_processes", default=8, type=int)
    parser.add_argument(
        "--upscale_ratio",
        default=2,
        type=int,
        choices=UltraZoom.AVAILABLE_UPSCALE_RATIOS,
    )
    parser.add_argument("--target_resolution", default=256, type=int)
    parser.add_argument("--min_gaussian_blur", default=0.0, type=float)
    parser.add_argument("--max_gaussian_blur", default=1.0, type=float)
    parser.add_argument("--min_gaussian_noise", default=0.0, type=float)
    parser.add_argument("--max_gaussian_noise", default=0.1, type=float)
    parser.add_argument("--min_compression", default=0.0, type=float)
    parser.add_argument("--max_compression", default=0.8, type=float)
    parser.add_argument("--brightness_jitter", default=0.1, type=float)
    parser.add_argument("--contrast_jitter", default=0.1, type=float)
    parser.add_argument("--saturation_jitter", default=0.1, type=float)
    parser.add_argument("--hue_jitter", default=0.1, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--upscaler_learning_rate", default=3e-4, type=float)
    parser.add_argument("--adaptive_loss_learning_rate", default=1e-3, type=float)
    parser.add_argument("--max_gradient_norm", default=2.0, type=float)
    parser.add_argument("--primary_channels", default=48, type=int)
    parser.add_argument("--primary_layers", default=4, type=int)
    parser.add_argument("--secondary_channels", default=96, type=int)
    parser.add_argument("--secondary_layers", default=4, type=int)
    parser.add_argument("--tertiary_channels", default=192, type=int)
    parser.add_argument("--tertiary_layers", default=4, type=int)
    parser.add_argument("--quaternary_channels", default=384, type=int)
    parser.add_argument("--quaternary_layers", default=8, type=int)
    parser.add_argument("--hidden_ratio", default=2, type=int)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--eval_interval", default=2, type=int)
    parser.add_argument("--checkpoint_interval", default=2, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"Batch size must be greater than 0, {args.batch_size} given.")

    if args.upscaler_learning_rate < 0:
        raise ValueError(
            f"Learning rate must be a positive value, {args.learning_rate} given."
        )

    if args.num_epochs < 1:
        raise ValueError(f"Must train for at least 1 epoch, {args.num_epochs} given.")

    if args.eval_interval < 1:
        raise ValueError(
            f"Eval interval must be greater than 0, {args.eval_interval} given."
        )

    if args.checkpoint_interval < 1:
        raise ValueError(
            f"Checkpoint interval must be greater than 0, {args.checkpoint_interval} given."
        )

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    if "mps" in args.device and not mps_is_available():
        raise RuntimeError("MPS is not available.")

    torch.backends.cudnn.conv.fp32_precision = "tf32"

    dtype = (
        torch.bfloat16
        if "cuda" in args.device and is_bf16_supported()
        else torch.float32
    )

    amp_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    logger = SummaryWriter(args.run_dir_path)

    new_dataset = partial(
        ImageFolder,
        target_resolution=args.target_resolution,
        upscale_ratio=args.upscale_ratio,
        min_gaussian_blur=args.min_gaussian_blur,
        max_gaussian_blur=args.max_gaussian_blur,
        min_gaussian_noise=args.min_gaussian_noise,
        max_gaussian_noise=args.max_gaussian_noise,
        min_compression=args.min_compression,
        max_compression=args.max_compression,
    )

    training = new_dataset(
        args.train_images_path,
        pre_transform=Compose(
            [
                RandomCrop(args.target_resolution),
                RandomHorizontalFlip(),
                ColorJitter(
                    brightness=args.brightness_jitter,
                    contrast=args.contrast_jitter,
                    hue=args.hue_jitter,
                    saturation=args.saturation_jitter,
                ),
            ]
        ),
    )

    testing = new_dataset(
        args.test_images_path,
        pre_transform=CenterCrop(args.target_resolution),
    )

    new_dataloader = partial(
        DataLoader,
        batch_size=args.batch_size,
        pin_memory="cuda" in args.device,
        num_workers=args.num_dataset_processes,
    )

    train_loader = new_dataloader(training, shuffle=True)
    test_loader = new_dataloader(testing)

    upscaler_args = {
        "upscale_ratio": args.upscale_ratio,
        "primary_channels": args.primary_channels,
        "primary_layers": args.primary_layers,
        "secondary_channels": args.secondary_channels,
        "secondary_layers": args.secondary_layers,
        "tertiary_channels": args.tertiary_channels,
        "tertiary_layers": args.tertiary_layers,
        "quaternary_channels": args.quaternary_channels,
        "quaternary_layers": args.quaternary_layers,
        "hidden_ratio": args.hidden_ratio,
        "degradation_features": 3,
    }

    upscaler = UltraZoom(**upscaler_args)

    upscaler.add_weight_norms()

    upscaler = torch.compile(upscaler)

    upscaler = upscaler.to(args.device)

    l2_loss_function = MSELoss()
    vgg_loss_function = VGGLoss()
    qa_loss_function = MSELoss()

    combined_loss_function = AdaptiveMultitaskLoss(4).to(args.device)

    vgg_loss_function = torch.compile(vgg_loss_function)

    vgg_loss_function = vgg_loss_function.to(args.device)

    print(f"Upscaler has {upscaler.num_trainable_params:,} trainable parameters")
    print(f"Perceptual loss function has {vgg_loss_function.num_params:,} parameters")

    upscaler_optimizer = AdamW(upscaler.parameters(), lr=args.upscaler_learning_rate)

    combined_loss_optimizer = SGD(
        combined_loss_function.parameters(), lr=args.adaptive_loss_learning_rate
    )

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)
    vif_metric = VisualInformationFidelity().to(args.device)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=True
        )

        upscaler.load_state_dict(checkpoint["upscaler"])
        upscaler_optimizer.load_state_dict(checkpoint["upscaler_optimizer"])

        combined_loss_function.load_state_dict(checkpoint["combined_loss"])
        combined_loss_optimizer.load_state_dict(checkpoint["combined_loss_optimizer"])

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    if args.activation_checkpointing:
        upscaler.encoder.enable_activation_checkpointing()

    print("Training ...")
    upscaler.train()

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_l2_loss, total_vgg22_loss, total_vgg54_loss = 0.0, 0.0, 0.0
        total_qa_loss, total_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0

        for step, (x, y_orig, y_deg) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y_orig = y_orig.to(args.device, non_blocking=True)
            y_deg = y_deg.to(args.device, non_blocking=True)

            with amp_context:
                y_pred_sr, y_pred_qa = upscaler.forward(x)

                l2_loss = l2_loss_function.forward(y_pred_sr, y_orig)
                vgg22_loss, vgg54_loss = vgg_loss_function.forward(y_pred_sr, y_orig)

                qa_loss = qa_loss_function.forward(y_pred_qa, y_deg)

                combined_loss = combined_loss_function.forward(
                    torch.stack([l2_loss, vgg22_loss, vgg54_loss, qa_loss])
                )

                scaled_loss = combined_loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                norm = clip_grad_norm_(upscaler.parameters(), args.max_gradient_norm)

                upscaler_optimizer.step()
                combined_loss_optimizer.step()

                upscaler_optimizer.zero_grad()
                combined_loss_optimizer.zero_grad()

                total_gradient_norm += norm.item()

                total_steps += 1

            total_l2_loss += l2_loss.item()
            total_vgg22_loss += vgg22_loss.item()
            total_vgg54_loss += vgg54_loss.item()
            total_qa_loss += qa_loss.item()

            total_batches += 1

        average_l2_loss = total_l2_loss / total_batches
        average_vgg22_loss = total_vgg22_loss / total_batches
        average_vgg54_loss = total_vgg54_loss / total_batches
        average_qa_loss = total_qa_loss / total_batches
        average_gradient_norm = total_gradient_norm / total_steps

        logger.add_scalar("Pixel L2", average_l2_loss, epoch)
        logger.add_scalar("VGG22 L2", average_vgg22_loss, epoch)
        logger.add_scalar("VGG54 L2", average_vgg54_loss, epoch)
        logger.add_scalar("QA L2", average_qa_loss, epoch)
        logger.add_scalar("Gradient Norm", average_gradient_norm, epoch)
        logger.add_scalar("Pixel Weight", combined_loss_function.loss_weights[0], epoch)
        logger.add_scalar("VGG22 Weight", combined_loss_function.loss_weights[1], epoch)
        logger.add_scalar("VGG54 Weight", combined_loss_function.loss_weights[2], epoch)
        logger.add_scalar("QA Weight", combined_loss_function.loss_weights[3], epoch)

        print(
            f"Epoch {epoch}:",
            f"Pixel L2: {average_l2_loss:.4},",
            f"VGG22 L2: {average_vgg22_loss:.4},",
            f"VGG54 L2: {average_vgg54_loss:.4},",
            f"QA L2: {average_qa_loss:.4},",
            f"Gradient Norm: {average_gradient_norm:.4},",
            f"Pixel Weight: {combined_loss_function.loss_weights[0]:.4},",
            f"VGG22 Weight: {combined_loss_function.loss_weights[1]:.4},",
            f"VGG54 Weight: {combined_loss_function.loss_weights[2]:.4},",
            f"QA Weight: {combined_loss_function.loss_weights[3]:.4}",
        )

        if epoch % args.eval_interval == 0:
            upscaler.eval()

            for x, y, _ in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                y_pred_sr = upscaler.upscale(x)

                psnr_metric.update(y_pred_sr, y)
                ssim_metric.update(y_pred_sr, y)
                vif_metric.update(y_pred_sr, y)

            psnr = psnr_metric.compute()
            ssim = ssim_metric.compute()
            vif = vif_metric.compute()

            logger.add_scalar("PSNR", psnr, epoch)
            logger.add_scalar("SSIM", ssim, epoch)
            logger.add_scalar("VIF", vif, epoch)

            print(
                f"PSNR: {psnr:.4},",
                f"SSIM: {ssim:.4},",
                f"VIF: {vif:.4}",
            )

            psnr_metric.reset()
            ssim_metric.reset()
            vif_metric.reset()

            upscaler.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "upscaler_args": upscaler_args,
                "upscaler": upscaler.state_dict(),
                "upscaler_optimizer": upscaler_optimizer.state_dict(),
                "combined_loss": combined_loss_function.state_dict(),
                "combined_loss_optimizer": combined_loss_optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")


if __name__ == "__main__":
    main()
