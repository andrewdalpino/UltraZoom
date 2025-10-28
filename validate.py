from time import time

from argparse import ArgumentParser

import torch

from torch.utils.data import DataLoader

from data import ImagePairs

from src.ultrazoom.model import UltraZoom

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    VisualInformationFidelity,
)

from tqdm import tqdm


def main():
    parser = ArgumentParser(
        description="Single-image super-resolution validation script"
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument(
        "--lr_images_path", default="./dataset/validate/Set14/LRbicx2", type=str
    )
    parser.add_argument(
        "--hr_images_path", default="./dataset/validate/Set14/GTmod12", type=str
    )
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

    model = UltraZoom(**checkpoint["model_args"])

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

    bicubic_psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    bicubic_ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)
    bicubic_vif_metric = VisualInformationFidelity().to(args.device)

    enhanced_psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    enhanced_ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)
    enhanced_vif_metric = VisualInformationFidelity().to(args.device)

    for x, y in tqdm(dataloader, desc="Testing", leave=False):
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)

        u_pred, u_bicubic = model.test_compare(x)

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
