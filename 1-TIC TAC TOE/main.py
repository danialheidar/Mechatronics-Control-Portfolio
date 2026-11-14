import torch

from train import train


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    train(
        iterations=100,
        games_per_iteration=100,
        simulations=500,
        batch_size=64,
        device=device
    )


if __name__ == "__main__":
    main()