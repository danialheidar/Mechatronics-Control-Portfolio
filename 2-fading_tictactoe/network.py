"""
network.py
==========

A small residual CNN with two heads (policy + value), in the AlphaZero
style, sized appropriately for a 3x3 game.

Input:  (batch, C, 3, 3) float tensor, C = config.input_planes (7 by
        default -- see ``game.FadingTicTacToe.to_tensor``).
Output: (policy_logits, value)
    policy_logits: (batch, 9) raw logits over the 9 board squares
                   (softmax + legal-move masking happens in mcts.py,
                   NOT inside the network).
    value:         (batch, 1) in [-1, 1] via tanh, representing the
                   predicted outcome from the perspective of the player
                   to move in the *input* state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class ResidualBlock(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class PolicyValueNet(nn.Module):
    """AlphaZero-style dual-head network for Fading-Move Tic-Tac-Toe."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        c = config.input_planes
        f = config.num_filters

        self.stem = nn.Sequential(
            nn.Conv2d(c, f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(f) for _ in range(config.num_res_blocks)]
        )

        # Policy head: 1x1 conv down to 2 channels, then a small MLP to
        # 9 logits (one per board square).
        self.policy_conv = nn.Sequential(
            nn.Conv2d(f, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Sequential(
            nn.Linear(2 * config.board_size * config.board_size, config.policy_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.policy_head_hidden, config.num_actions),
        )

        # Value head: 1x1 conv down to 1 channel, then a small MLP to a
        # single scalar, squashed to [-1, 1] with tanh.
        self.value_conv = nn.Sequential(
            nn.Conv2d(f, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(config.board_size * config.board_size, config.value_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.value_head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.res_blocks(x)

        p = self.policy_conv(x)
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        v = self.value_conv(x)
        v = v.flatten(start_dim=1)
        v = self.value_fc(v)
        value = torch.tanh(v)

        return policy_logits, value

    @torch.no_grad()
    def predict(self, state_tensor: np.ndarray, device: torch.device) -> Tuple[np.ndarray, float]:
        """Convenience single-state inference used by MCTS.

        Args:
            state_tensor: numpy array of shape (C, 3, 3).
            device: torch device to run inference on.

        Returns:
            (policy_probs, value): policy_probs is a numpy array of
            shape (9,) of softmax probabilities over *all* 9 actions
            (legal-move masking is the caller's responsibility); value
            is a Python float in [-1, 1].
        """
        self.eval()
        t = torch.from_numpy(state_tensor).unsqueeze(0).to(device=device, dtype=torch.float32)
        logits, value = self.forward(t)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs.astype(np.float32), float(value.item())
