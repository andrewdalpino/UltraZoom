from time import time

from argparse import ArgumentParser

import torch

from torchvision.io import decode_image
from torchvision.transforms.v2 import ToDtype
from torchvision.utils import make_grid, save_image

from src.ultrazoom.model import UltraZoom

import matplotlib.pyplot as plt

from src.ultrazoom.control import ControlVector


def main():
    parser = ArgumentParser(description="Super-resolution upscaling script")

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--gaussian_blur", default=0.5, type=float)
    parser.add_argument("--gaussian_noise", default=0.5, type=float)
    parser.add_argument("--jpeg_compression", default=0.5, type=float)
    parser.add_argument("--device", default="cpu", type=str)

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available.")

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

    image_to_tensor = ToDtype(torch.float32, scale=True)

    image = decode_image(args.image_path, mode="RGB")

    x = image_to_tensor(image).unsqueeze(0).to(args.device)

    c = (
        ControlVector(
            gaussian_blur=args.gaussian_blur,
            gaussian_noise=args.gaussian_noise,
            jpeg_compression=args.jpeg_compression,
        )
        .to_tensor()
        .to(args.device)
        .unsqueeze(0)
    )

    print("Upscaling ...")

    with torch.inference_mode():
        y_pred, y_bicubic = model.forward(x, c)

    y_pred = torch.clamp(y_pred, 0, 1)
    y_bicubic = torch.clamp(y_bicubic, 0, 1)

    pair = torch.stack(
        [
            y_bicubic.squeeze(0),
            y_pred.squeeze(0),
        ],
        dim=0,
    )

    grid = make_grid(pair, nrow=2)

    grid = grid.permute(1, 2, 0).to("cpu")

    plt.imshow(grid)
    plt.show()

    if "y" in input("Save image? (yes|no): ").lower():
        filename = f"out_{time()}"

        save_image(y_pred.squeeze(0), f"{filename}.png")


if __name__ == "__main__":
    main()
