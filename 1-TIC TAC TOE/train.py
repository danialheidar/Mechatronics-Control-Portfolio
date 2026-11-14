import random

import numpy as np
import torch
import torch.nn.functional as F

from network import AlphaZeroNet
from self_play import self_play


def train_step(
    network,
    optimizer,
    batch,
    device="cpu"
):
    states = torch.tensor(
        np.array([x[0] for x in batch]),
        dtype=torch.float32,
        device=device
    )

    target_policies = torch.tensor(
        np.array([x[1] for x in batch]),
        dtype=torch.float32,
        device=device
    )

    target_values = torch.tensor(
        np.array([x[2] for x in batch]),
        dtype=torch.float32,
        device=device
    ).unsqueeze(1)

    policy_logits, values = network(states)

    # Policy loss
    policy_loss = -torch.mean(
        torch.sum(
            target_policies
            * F.log_softmax(policy_logits, dim=1),
            dim=1
        )
    )

    # Value loss
    value_loss = F.mse_loss(
        values,
        target_values
    )

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(
    iterations=100,
    games_per_iteration=100,
    simulations=500,
    batch_size=64,
    device="cpu"
):
    network = AlphaZeroNet().to(device)

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=0.001
    )

    replay_buffer = []

    for iteration in range(iterations):

        print(
            f"\n===== Iteration "
            f"{iteration + 1}/{iterations} ====="
        )

        # Self-play
        for game_number in range(games_per_iteration):

            data = self_play(
                network,
                simulations=simulations,
                temperature=1.0,
                device=device
            )

            replay_buffer.extend(data)

            print(
                f"Self-play game "
                f"{game_number + 1}/{games_per_iteration}"
            )

        # محدود کردن replay buffer
        replay_buffer = replay_buffer[-10000:]

        # Training
        if len(replay_buffer) >= batch_size:

            steps = min(
                100,
                len(replay_buffer) // batch_size
            )

            losses = []

            for _ in range(steps):

                batch = random.sample(
                    replay_buffer,
                    batch_size
                )

                loss = train_step(
                    network,
                    optimizer,
                    batch,
                    device
                )

                losses.append(loss)

            print(
                f"Training loss: "
                f"{np.mean(losses):.4f}"
            )

        # ذخیره مدل
        torch.save(
            network.state_dict(),
            "alphazero_tictactoe.pth"
        )

        print("Model saved.")

    return network