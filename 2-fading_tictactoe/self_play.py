"""
self_play.py
============

Generates self-play training data: both sides are played by the same
network-guided MCTS. For every move we record:

    (state_tensor, mcts_visit_policy, player_to_move)

Once the game ends, each stored position is converted into a full
training example ``(state_tensor, policy_target, value_target)`` where
``value_target`` is the game outcome (+1 win / -1 loss / 0 draw) from
the perspective of the player who was to move in that stored state --
*not* necessarily the perspective of whoever eventually won.

A temperature schedule controls exploration: early moves sample from
the (temperature-scaled) visit-count distribution, later moves are
played (near-)greedily, per ``config.temperature_moves``.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch

from config import Config
from game import FadingTicTacToe
from mcts import MCTS
from network import PolicyValueNet
from symmetry import all_symmetries

Example = Tuple[np.ndarray, np.ndarray, float]  # (state, policy, value)


def play_one_game(
    network: PolicyValueNet,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
) -> List[Example]:
    """Play a single self-play game and return training examples with
    correctly assigned value targets (one example per ply)."""
    game = FadingTicTacToe(config)
    mcts = MCTS(network, config, device, rng)

    # (state_tensor, visit_policy, player_to_move_at_this_state)
    trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []
    move_idx = 0

    while not game.is_terminal():
        root = mcts.run(game, add_dirichlet_noise=True)
        visit_policy = MCTS.visit_count_policy(root, config.num_actions)

        temperature = (
            config.temperature_early
            if move_idx < config.temperature_moves
            else config.temperature_late
        )
        action = MCTS.select_action(visit_policy, temperature, rng)

        trajectory.append((game.to_tensor(), visit_policy, game.current_player))
        game.make_move(action)
        move_idx += 1

    winner = game.winner()  # 1, -1, or 0
    examples: List[Example] = []
    for state_tensor, policy_target, player_to_move in trajectory:
        if winner == 0:
            value_target = 0.0
        else:
            value_target = 1.0 if winner == player_to_move else -1.0
        examples.append((state_tensor, policy_target, value_target))
    return examples


def augment_examples(examples: List[Example], config: Config) -> List[Example]:
    """Expand each example into all 8 board symmetries (or return the
    examples unchanged if augmentation is disabled)."""
    if not config.use_symmetry_augmentation:
        return examples
    augmented: List[Example] = []
    for state, policy, value in examples:
        for sym_state, sym_policy in all_symmetries(state, policy):
            augmented.append((sym_state, sym_policy, value))
    return augmented


def generate_self_play_data(
    network: PolicyValueNet,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
    num_games: int,
) -> List[Example]:
    """Play ``num_games`` self-play games and return augmented examples."""
    all_examples: List[Example] = []
    for _ in range(num_games):
        game_examples = play_one_game(network, config, device, rng)
        all_examples.extend(augment_examples(game_examples, config))
    return all_examples


class ReplayBuffer:
    """A simple fixed-capacity FIFO replay buffer of training examples."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: Deque[Example] = deque(maxlen=capacity)

    def add_many(self, examples: List[Example]) -> None:
        self._data.extend(examples)

    def __len__(self) -> int:
        return len(self._data)

    def sample(self, batch_size: int, rng: random.Random) -> List[Example]:
        n = min(batch_size, len(self._data))
        return rng.sample(list(self._data), n)
