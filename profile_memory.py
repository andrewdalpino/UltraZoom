import torch
import torch.profiler

import sys
import os

from src.ultrazoom.model import UltraZoom

# Add the src directory to the path to import UltraZoom
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def profile_ultrazoom():
    """Profile the UltraZoom model using PyTorch's native profiler."""

    # Create UltraZoom model with typical configuration
    model = UltraZoom(
        upscale_ratio=4,
        primary_channels=64,
        primary_layers=6,
        secondary_channels=128,
        secondary_layers=12,
        tertiary_channels=256,
        tertiary_layers=24,
        quaternary_channels=512,
        quaternary_layers=12,
        hidden_ratio=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Model has {model.num_params:,} parameters")
    print(f"Using device: {device}")

    # Create dummy image data (B, C, H, W) - typical super-resolution input
    batch_size = 1
    input_height, input_width = 256, 256
    input_tensor = torch.randn(batch_size, 3, input_height, input_width).to(device)

    # For training, we also need a target (high-resolution) image
    target_height = input_height * model.upscale_ratio
    target_width = input_width * model.upscale_ratio
    target_tensor = torch.randn(batch_size, 3, target_height, target_width).to(device)

    # Loss function and optimizer
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Target shape: {target_tensor.shape}")

    # Configure PyTorch profiler with web interface
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=3, warmup=2, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        use_cuda=torch.cuda.is_available(),
    ) as profiler:

        # Training loop
        for step in range(20):
            optimizer.zero_grad()

            # Forward pass
            output = model(input_tensor)
            loss = criterion(output, target_tensor)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Step the profiler
            profiler.step()

            if step == 0:
                print(f"Output shape: {output.shape}")
                print(f"Initial loss: {loss.item():.4f}")

    print("Profiling complete!")
    print("To view results:")
    print("1. Install tensorboard: pip install tensorboard")
    print("2. Run: tensorboard --logdir=./profiler_logs")
    print("3. Open http://localhost:6006 in your browser")
    print("4. Click on the 'Profile' tab to view profiling data")


if __name__ == "__main__":
    profile_ultrazoom()
