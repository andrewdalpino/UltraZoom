import random

from functools import partial

from argparse import ArgumentParser

import torch

from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
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

from torchmetrics.classification import BinaryPrecision, BinaryRecall

from data import ImageFolder
from src.ultrazoom.model import UltraZoom, Bouncer
from loss import RelativisticBCELoss

from tqdm import tqdm


def main():
    parser = ArgumentParser(description="Generative adversarial fine-tuning script.")

    parser.add_argument("--base_checkpoint_path", type=str, required=True)
    parser.add_argument("--train_images_path", default="./dataset/train", type=str)
    parser.add_argument("--test_images_path", default="./dataset/test", type=str)
    parser.add_argument("--num_dataset_processes", default=8, type=int)
    parser.add_argument("--target_resolution", default=256, type=int)
    parser.add_argument("--blur_amount", default=0.5, type=float)
    parser.add_argument("--noise_amount", default=0.02, type=float)
    parser.add_argument("--min_compression", default=0.1, type=float)
    parser.add_argument("--max_compression", default=0.3, type=float)
    parser.add_argument("--brightness_jitter", default=0.1, type=float)
    parser.add_argument("--contrast_jitter", default=0.1, type=float)
    parser.add_argument("--saturation_jitter", default=0.1, type=float)
    parser.add_argument("--hue_jitter", default=0.1, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--upscaler_learning_rate", default=1e-4, type=float)
    parser.add_argument("--upscaler_max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--critic_learning_rate", default=2e-4, type=float)
    parser.add_argument("--critic_max_gradient_norm", default=2.0, type=float)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--critic_warmup_epochs", default=4, type=int)
    parser.add_argument(
        "--critic_model_size", default="small", choices=Bouncer.AVAILABLE_MODEL_SIZES
    )
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

    if args.upscaler_learning_rate < 0 or args.critic_learning_rate < 0:
        raise ValueError(f"Learning rate must be a positive value.")

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

    torch.set_float32_matmul_precision("high")

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

    checkpoint = torch.load(
        args.base_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    upscaler_args = checkpoint["model_args"]

    new_dataset = partial(
        ImageFolder,
        target_resolution=args.target_resolution,
        upscale_ratio=upscaler_args["upscale_ratio"],
        blur_amount=args.blur_amount,
        noise_amount=args.noise_amount,
        min_compression=args.min_compression,
        max_compression=args.max_compression,
    )

    training = new_dataset(
        args.train_images_path,
        pre_transformer=Compose(
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
        pre_transformer=CenterCrop(args.target_resolution),
    )

    new_dataloader = partial(
        DataLoader,
        batch_size=args.batch_size,
        pin_memory="cuda" in args.device,
        num_workers=args.num_dataset_processes,
    )

    train_loader = new_dataloader(training, shuffle=True)
    test_loader = new_dataloader(testing)

    upscaler = UltraZoom(**upscaler_args)

    upscaler.add_weight_norms()

    state_dict = checkpoint["model"]

    # Compensate for compiled state dict.
    for key in list(state_dict.keys()):
        state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)

    upscaler.load_state_dict(state_dict)

    upscaler = upscaler.to(args.device)

    print("Base model loaded successfully")

    critic_args = {
        "model_size": args.critic_model_size,
    }

    critic = Bouncer.from_preconfigured(**critic_args)

    critic.add_spectral_norms()

    critic = critic.to(args.device)

    pixel_l2_loss = MSELoss()
    stage_1_l2_loss = MSELoss()
    bce_loss = RelativisticBCELoss()

    upscaler_optimizer = AdamW(upscaler.parameters(), lr=args.upscaler_learning_rate)
    critic_optimizer = AdamW(critic.parameters(), lr=args.critic_learning_rate)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=True
        )

        upscaler.load_state_dict(checkpoint["model"])
        upscaler_optimizer.load_state_dict(checkpoint["model_optimizer"])

        critic.load_state_dict(checkpoint["critic"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    if args.activation_checkpointing:
        upscaler.encoder.enable_activation_checkpointing()
        critic.detector.enable_activation_checkpointing()

    print(f"Upscaler has {upscaler.num_trainable_params:,} trainable parameters")
    print(f"Critic has {critic.num_trainable_params:,} trainable parameters")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)
    vif_metric = VisualInformationFidelity().to(args.device)

    precision_metric = BinaryPrecision().to(args.device)
    recall_metric = BinaryRecall().to(args.device)

    print("Fine-tuning ...")

    upscaler.train()
    critic.train()

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_pixel_l2, total_stage_1_l2 = 0.0, 0.0
        total_u_bce, total_c_bce = 0.0, 0.0
        total_u_gradient_norm, total_c_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0

        is_warmup = epoch <= args.critic_warmup_epochs

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            y_real = torch.full((y.size(0), 1), 1.0).to(args.device)
            y_fake = torch.full((y.size(0), 1), 0.0).to(args.device)

            update_this_step = step % args.gradient_accumulation_steps == 0

            with amp_context:
                u_pred, _ = upscaler.forward(x)

                _, _, _, _, c_pred_fake = critic.forward(u_pred.detach())
                _, _, _, _, c_pred_real = critic.forward(y)

                c_bce = bce_loss(c_pred_real, c_pred_fake, y_real, y_fake)

                scaled_c_loss = c_bce / args.gradient_accumulation_steps

            scaled_c_loss.backward()

            if update_this_step:
                c_norm = clip_grad_norm_(
                    critic.parameters(), args.critic_max_gradient_norm
                )

                critic_optimizer.step()

                critic_optimizer.zero_grad()

                total_c_gradient_norm += c_norm.item()

                total_steps += 1

            total_c_bce += c_bce.item()

            if not is_warmup:
                with amp_context:
                    pixel_l2 = pixel_l2_loss(u_pred, y)

                    z1_fake, _, _, _, c_pred_fake = critic.forward(u_pred)
                    z1_real, _, _, _, c_pred_real = critic.forward(y)

                    stage_1_l2 = stage_1_l2_loss(z1_fake, z1_real)

                    u_bce = bce_loss(c_pred_real, c_pred_fake, y_fake, y_real)

                    combined_u_loss = (
                        pixel_l2 / pixel_l2.detach()
                        + stage_1_l2 / stage_1_l2.detach()
                        + u_bce / u_bce.detach()
                    )

                    scaled_u_loss = combined_u_loss / args.gradient_accumulation_steps

                scaled_u_loss.backward()

                if update_this_step:
                    u_norm = clip_grad_norm_(
                        upscaler.parameters(), args.upscaler_max_gradient_norm
                    )

                    upscaler_optimizer.step()

                    upscaler_optimizer.zero_grad()

                    total_u_gradient_norm += u_norm.item()

                total_pixel_l2 += pixel_l2.item()
                total_stage_1_l2 += stage_1_l2.item()
                total_u_bce += u_bce.item()

            total_batches += 1

        average_pixel_l2 = total_pixel_l2 / total_batches
        average_stage_1_l2 = total_stage_1_l2 / total_batches
        average_u_bce = total_u_bce / total_batches
        average_c_bce = total_c_bce / total_batches

        average_u_gradient_norm = total_u_gradient_norm / total_steps
        average_c_gradient_norm = total_c_gradient_norm / total_steps

        logger.add_scalar("Pixel L2", average_pixel_l2, epoch)
        logger.add_scalar("Stage 1 L2", average_stage_1_l2, epoch)
        logger.add_scalar("Upscaler BCE", average_u_bce, epoch)
        logger.add_scalar("Upscaler Norm", average_u_gradient_norm, epoch)
        logger.add_scalar("Critic BCE", average_c_bce, epoch)
        logger.add_scalar("Critic Norm", average_c_gradient_norm, epoch)

        print(
            f"Epoch {epoch}:",
            f"Pixel L2: {average_pixel_l2:.5},",
            f"Stage 1 L2: {average_stage_1_l2:.5},",
            f"Upscaler BCE: {average_u_bce:.5},",
            f"Upscaler Norm: {average_u_gradient_norm:.4},",
            f"Critic BCE: {average_c_bce:.5},",
            f"Critic Norm: {average_c_gradient_norm:.4}",
        )

        if epoch % args.eval_interval == 0:
            upscaler.eval()
            critic.eval()

            for x, y in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                y_real = torch.full((y.size(0), 1), 1.0).to(args.device)
                y_fake = torch.full((y.size(0), 1), 0.0).to(args.device)

                u_pred = upscaler.upscale(x)

                c_pred_real = critic.predict(y)
                c_pred_fake = critic.predict(u_pred)

                c_pred_real -= c_pred_fake.mean()
                c_pred_fake -= c_pred_real.mean()

                c_pred = torch.cat((c_pred_real, c_pred_fake), dim=0)
                labels = torch.cat((y_real, y_fake), dim=0)

                psnr_metric.update(u_pred, y)
                ssim_metric.update(u_pred, y)
                vif_metric.update(u_pred, y)

                precision_metric.update(c_pred, labels)
                recall_metric.update(c_pred, labels)

            psnr = psnr_metric.compute()
            ssim = ssim_metric.compute()
            vif = vif_metric.compute()

            precision = precision_metric.compute()
            recall = recall_metric.compute()

            if precision + recall != 0:
                f1_score = (2 * precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            logger.add_scalar("PSNR", psnr, epoch)
            logger.add_scalar("SSIM", ssim, epoch)
            logger.add_scalar("VIF", vif, epoch)
            logger.add_scalar("F1 Score", f1_score, epoch)
            logger.add_scalar("Precision", precision, epoch)
            logger.add_scalar("Recall", recall, epoch)

            print(
                f"PSNR: {psnr:.5},",
                f"SSIM: {ssim:.5},",
                f"VIF: {vif:.5},",
                f"F1 Score: {f1_score:.5},",
                f"Precision: {precision:.5},",
                f"Recall: {recall:.5}",
            )

            psnr_metric.reset()
            ssim_metric.reset()
            vif_metric.reset()

            precision_metric.reset()
            recall_metric.reset()

            upscaler.train()
            critic.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_args": upscaler_args,
                "model": upscaler.state_dict(),
                "model_optimizer": upscaler_optimizer.state_dict(),
                "critic_args": critic_args,
                "critic": critic.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")


if __name__ == "__main__":
    main()
