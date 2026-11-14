"""
train.py
========

Two things live here:

1. Checkpoint save/load utilities (shared by evaluate.py and play.py).
2. The AlphaZero-style iterative-improvement training loop:

    for each iteration:
        - self-play a batch of games with the current *best* network
          to generate training data (appended to a replay buffer),
        - train a *candidate* network (continuing from the best
          network's weights) on samples drawn from the buffer,
        - evaluate the candidate against the best network,
        - promote the candidate to "best" iff it wins a sufficient
          fraction of decisive evaluation games.

Run as a script to train from scratch or resume:

    python train.py --iterations 20 --games-per-iter 50
    python train.py --resume checkpoints/best.pt --iterations 10
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from network import PolicyValueNet
from self_play import ReplayBuffer, generate_self_play_data, Example
from evaluate import evaluate_networks, RandomAgent, NetworkAgent


# ----------------------------------------------------------------------
# Reproducibility / device
# ----------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: Config) -> torch.device:
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if config.device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(config.device)


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------
def save_checkpoint(
    path: str,
    network: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    config: Config,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
            "config": config.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str, device: torch.device
) -> Tuple[PolicyValueNet, torch.optim.Optimizer, int, Config]:
    checkpoint = torch.load(path, map_location=device)
    config = Config.from_dict(checkpoint["config"])
    network = PolicyValueNet(config).to(device)
    network.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return network, optimizer, iteration, config


# ----------------------------------------------------------------------
# Training step
# ----------------------------------------------------------------------
def compute_loss(
    network: PolicyValueNet,
    states: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    config: Config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AlphaZero-style combined loss.

    Policy loss: cross-entropy between the network's policy distribution
    and the MCTS visit-count targets (a soft-label cross entropy, since
    the targets are full distributions rather than one-hot labels).
    Value loss: MSE between predicted and target scalar outcomes.
    L2 regularization is handled via the optimizer's ``weight_decay``.
    """
    policy_logits, value_pred = network(states)
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets)
    total_loss = (
        config.policy_loss_weight * policy_loss + config.value_loss_weight * value_loss
    )
    return total_loss, policy_loss, value_loss


def train_on_buffer(
    network: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    config: Config,
    device: torch.device,
    rng: random.Random,
) -> None:
    """Run ``config.train_epochs_per_iteration`` epochs of minibatch SGD
    over (samples drawn from) the replay buffer."""
    network.train()
    num_batches_per_epoch = max(1, len(buffer) // config.batch_size)

    for epoch in range(config.train_epochs_per_iteration):
        epoch_loss = 0.0
        for _ in range(num_batches_per_epoch):
            batch = buffer.sample(config.batch_size, rng)
            states = torch.from_numpy(np.stack([b[0] for b in batch])).to(
                device=device, dtype=torch.float32
            )
            policies = torch.from_numpy(np.stack([b[1] for b in batch])).to(
                device=device, dtype=torch.float32
            )
            values = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            loss, policy_loss, value_loss = compute_loss(network, states, policies, values, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches_per_epoch
        print(f"    epoch {epoch + 1}/{config.train_epochs_per_iteration}  "
              f"avg_loss={avg_loss:.4f}")


# ----------------------------------------------------------------------
# Iterative improvement loop
# ----------------------------------------------------------------------
def run_training(
    config: Config,
    num_iterations: Optional[int] = None,
    resume_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> None:
    set_seed(config.seed)
    device = get_device(config)
    checkpoint_dir = checkpoint_dir or config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    py_rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    start_iteration = 0
    if resume_path is not None and os.path.exists(resume_path):
        best_net, best_optimizer, start_iteration, loaded_config = load_checkpoint(
            resume_path, device
        )
        config = loaded_config
        print(f"Resumed from {resume_path} at iteration {start_iteration}.")
    else:
        best_net = PolicyValueNet(config).to(device)
        best_optimizer = torch.optim.Adam(
            best_net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    buffer = ReplayBuffer(config.replay_buffer_size)
    total_iterations = num_iterations if num_iterations is not None else config.num_iterations

    for it in range(start_iteration, start_iteration + total_iterations):
        t0 = time.time()
        print(f"=== Iteration {it + 1} ===")

        # 1) Self-play with the current best network.
        best_net.eval()
        new_examples = generate_self_play_data(
            best_net, config, device, np_rng, config.num_self_play_games
        )
        buffer.add_many(new_examples)
        print(f"  self-play: {config.num_self_play_games} games -> "
              f"{len(new_examples)} examples (buffer size {len(buffer)})")

        if len(buffer) < config.min_buffer_size_to_train:
            print("  buffer below min size, skipping training this iteration.")
            save_checkpoint(
                os.path.join(checkpoint_dir, "best.pt"), best_net, best_optimizer, it + 1, config
            )
            continue

        # 2) Train a candidate network, continuing from the best weights.
        candidate_net = copy.deepcopy(best_net).to(device)
        candidate_optimizer = torch.optim.Adam(
            candidate_net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        candidate_optimizer.load_state_dict(best_optimizer.state_dict())
        train_on_buffer(candidate_net, candidate_optimizer, buffer, config, device, py_rng)

        # 3) Evaluate candidate vs. best.
        candidate_agent = NetworkAgent(candidate_net, config, device, num_simulations=config.eval_num_simulations)
        best_agent = NetworkAgent(best_net, config, device, num_simulations=config.eval_num_simulations)
        results = evaluate_networks(candidate_agent, best_agent, config.eval_games, np_rng)
        decisive = results.wins + results.losses
        win_rate = results.wins / decisive if decisive > 0 else 0.0
        print(f"  eval vs best: {results.wins}W {results.losses}L {results.draws}D "
              f"(decisive win rate {win_rate:.2f})")

        # 4) Promote candidate if it clears the acceptance bar.
        if decisive >= config.min_decisive_games and win_rate >= config.acceptance_win_rate:
            best_net = candidate_net
            best_optimizer = candidate_optimizer
            print("  candidate ACCEPTED as new best network.")
        else:
            print("  candidate rejected; keeping previous best network.")

        save_checkpoint(
            os.path.join(checkpoint_dir, "best.pt"), best_net, best_optimizer, it + 1, config
        )
        save_checkpoint(
            os.path.join(checkpoint_dir, f"iter_{it + 1:04d}.pt"),
            best_net,
            best_optimizer,
            it + 1,
            config,
        )
        print(f"  iteration took {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an AlphaZero agent for Fading-Move Tic-Tac-Toe.")
    parser.add_argument("--iterations", type=int, default=None, help="Number of training iterations to run.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save checkpoints.")
    parser.add_argument("--games-per-iter", type=int, default=None, help="Override self-play games per iteration.")
    parser.add_argument("--simulations", type=int, default=None, help="Override MCTS simulations per move.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    args = parser.parse_args()

    config = Config()
    if args.games_per_iter is not None:
        config.num_self_play_games = args.games_per_iter
    if args.simulations is not None:
        config.num_simulations = args.simulations
    if args.seed is not None:
        config.seed = args.seed

    run_training(
        config,
        num_iterations=args.iterations,
        resume_path=args.resume,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
