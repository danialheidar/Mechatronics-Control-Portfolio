"""
config.py
=========

Central configuration object for the whole project.

Every tunable knob used by the game engine, MCTS, self-play, training,
evaluation and the CLI lives here so that no magic numbers are scattered
around the codebase.  A `Config` instance is also saved inside every
checkpoint so that training runs are reproducible and self-describing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import json


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed: int = 42

    # ------------------------------------------------------------------
    # Game rules
    # ------------------------------------------------------------------
    # Maximum number of *active* marks a single player may have on the
    # board at once (this is the "fading" window described in the spec).
    max_active_marks: int = 3
    # Safety valve: if a single game somehow exceeds this many plies
    # without terminating (should never happen because repetition
    # detection below always fires first), it is scored as a draw.
    max_game_length: int = 200
    # A given (history_X, history_O, current_player) state signature is
    # declared a draw once it has been seen this many times in one game.
    # This is what makes the fading mechanic's potential cycles provably
    # finite in practice.
    repetition_draw_count: int = 3

    # ------------------------------------------------------------------
    # Neural network
    # ------------------------------------------------------------------
    input_planes: int = 7          # see network.py / game.py docstrings
    board_size: int = 3
    num_actions: int = 9
    num_res_blocks: int = 4
    num_filters: int = 64
    value_head_hidden: int = 32
    policy_head_hidden: int = 32

    # ------------------------------------------------------------------
    # MCTS
    # ------------------------------------------------------------------
    num_simulations: int = 1000
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.6
    dirichlet_epsilon: float = 0.25  # weight of Dirichlet noise at root

    # ------------------------------------------------------------------
    # Self-play / exploration
    # ------------------------------------------------------------------
    num_self_play_games: int = 200
    # Moves with ply index < temperature_moves are sampled from the
    # (temperature-scaled) visit-count distribution; afterwards play is
    # (near-)greedy. This is the "temperature schedule".
    temperature_moves: int = 8
    temperature_early: float = 1.0
    temperature_late: float = 0.05

    # ------------------------------------------------------------------
    # Replay buffer / training
    # ------------------------------------------------------------------
    replay_buffer_size: int = 50_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_epochs_per_iteration: int = 4
    min_buffer_size_to_train: int = 512
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    grad_clip_norm: float = 5.0

    # ------------------------------------------------------------------
    # Iterative improvement loop
    # ------------------------------------------------------------------
    num_iterations: int = 20
    eval_games: int = 40
    eval_num_simulations: int = 100
    # New network replaces "best" network if it wins at least this
    # fraction of decisive (non-draw) evaluation games.
    acceptance_win_rate: float = 0.55
    # Fallback: if there are too few decisive games to be meaningful,
    # require at least this many decisive games before accepting.
    min_decisive_games: int = 6

    # ------------------------------------------------------------------
    # Symmetry augmentation
    # ------------------------------------------------------------------
    use_symmetry_augmentation: bool = True

    # ------------------------------------------------------------------
    # Misc / bookkeeping
    # ------------------------------------------------------------------
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"  # overridden to "cuda" automatically if available

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        return Config(**d)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r") as f:
            d = json.load(f)
        return Config.from_dict(d)
