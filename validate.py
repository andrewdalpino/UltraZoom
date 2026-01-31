from time import time

from argparse import ArgumentParser

import torch

from torch.utils.data import DataLoader

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    VisualInformationFidelity,
)

from src.ultrazoom.model import MewZoom
from src.ultrazoom.control import ControlVector

from data import ImagePairs

from tqdm import tqdm


def main():
    parser = ArgumentParser(
        description="Single-image super-resolution validation script"
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--lr_images_path", default="./dataset/validate/lr", type=str)
    parser.add_argument("--hr_images_path", default="./dataset/validate/hr", type=str)
    parser.add_argument("--gaussian_blur", default=0.1, type=float)
    parser.add_argument("--gaussian_noise", default=0.1, type=float)
    parser.add_argument("--jpeg_compression", default=0.1, type=float)
    parser.add_argument("--device", default="cpu", type=str)

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available.")

    dataset = ImagePairs(args.lr_images_path, args.hr_images_path)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory="cuda" in args.device,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)

    model = MewZoom(**checkpoint["model_args"])

    model.add_weight_norms()

    state_dict = checkpoint["model"]

    # Compensate for compiled state dict.
    for key in list(state_dict.keys()):
        state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)

    model.load_state_dict(state_dict)

    model.remove_parameterizations()

    model = model.to(args.device)

    model.eval()

    print("Model checkpoint loaded successfully")

    c_hat = (
        ControlVector(
            gaussian_blur=args.gaussian_blur,
            gaussian_noise=args.gaussian_noise,
            jpeg_compression=args.jpeg_compression,
        )
        .to_tensor()
        .to(args.device)
        .unsqueeze(0)
    )

    bicubic_psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    bicubic_ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)
    bicubic_vif_metric = VisualInformationFidelity().to(args.device)

    enhanced_psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    enhanced_ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)
    enhanced_vif_metric = VisualInformationFidelity().to(args.device)

    for x, y in tqdm(dataloader, desc="Testing", leave=False):
        x = x.to(args.device, non_blocking=True)
        c = c_hat.repeat(x.size(0), 1)
        y = y.to(args.device, non_blocking=True)

        u_pred, u_bicubic = model.test_compare(x, c)

        bicubic_psnr_metric.update(u_bicubic, y)
        bicubic_ssim_metric.update(u_bicubic, y)
        bicubic_vif_metric.update(u_bicubic, y)

        enhanced_psnr_metric.update(u_pred, y)
        enhanced_ssim_metric.update(u_pred, y)
        enhanced_vif_metric.update(u_pred, y)

    bicubic_psnr = bicubic_psnr_metric.compute()
    bicubic_ssim = bicubic_ssim_metric.compute()
    bicubic_vif = bicubic_vif_metric.compute()

    enhanced_psnr = enhanced_psnr_metric.compute()
    enhanced_ssim = enhanced_ssim_metric.compute()
    enhanced_vif = enhanced_vif_metric.compute()

    print(
        f"Bicubic PSNR: {bicubic_psnr:.5f}, "
        f"Bicubic SSIM: {bicubic_ssim:.5f}, "
        f"Bicubic VIF: {bicubic_vif:.5f}"
    )

    print(
        f"Enhanced PSNR: {enhanced_psnr:.5f}, "
        f"Enhanced SSIM: {enhanced_ssim:.5f}, "
        f"Enhanced VIF: {enhanced_vif:.5f}"
    )


if __name__ == "__main__":
    main()
