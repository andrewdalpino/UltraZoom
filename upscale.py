from argparse import ArgumentParser

import torch

from torchvision.io import decode_image
from torchvision.transforms.v2 import ToDtype
from torchvision.utils import make_grid

from model import UltraZoom

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("qt5agg")


def main():
    parser = ArgumentParser(description="Super-resolution upscaling script")

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available.")

    checkpoint = torch.load(
        args.checkpoint_path, map_location=args.device, weights_only=True
    )

    model = UltraZoom(**checkpoint["model_args"])

    model.add_weight_norms()

    if args.device == "cuda":
        print("Compiling model")

        model = torch.compile(model)

    model = model.to(args.device)

    model.load_state_dict(checkpoint["model"])

    model.remove_weight_norms()

    model.eval()

    print("Model checkpoint loaded successfully")

    image_to_tensor = ToDtype(torch.float32, scale=True)

    image = decode_image(args.image_path, mode="RGB")

    x = image_to_tensor(image).unsqueeze(0).to(args.device)

    print("Upscaling ...")
    y_pred, y_bicubic = model.test_compare(x)

    pair = torch.stack(
        [
            y_bicubic.squeeze(0),
            y_pred.squeeze(0),
        ],
        dim=0,
    )

    pair = pair.to("cpu")

    grid = make_grid(pair, nrow=2)

    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
