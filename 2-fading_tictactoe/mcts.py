"""
mcts.py
=======

AlphaZero-style Monte Carlo Tree Search guided by a policy/value
network, using PUCT for action selection.

Value-perspective convention (read this before touching backup logic)
-----------------------------------------------------------------------
Every ``Node`` corresponds to a game state, and stores ``to_play``: the
player whose turn it is *at that state*. The network's value output,
and every value we store/propagate, is always "from the perspective of
the player to move at that node".

Because ``FadingTicTacToe.make_move`` unconditionally flips
``current_player`` after every move (even a game-ending one -- see the
comment in ``game.py``), ``to_play`` strictly alternates between a
parent node and each of its children, with **no exceptions**,
including at terminal leaves. This means a single sign flip per tree
level, applied while walking back up the search path, is always
correct: a value ``v`` that is good for the player at depth ``d`` is
exactly as good as ``-v`` is bad for the player at depth ``d-1``.

Terminal leaves are handled without ever calling the network: the
ground-truth game result is converted directly into a value from the
perspective of the player to move at that (post-terminal) node.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import Config
from game import FadingTicTacToe
from network import PolicyValueNet


class Node:
    __slots__ = ("prior", "visit_count", "total_value", "children", "to_play", "expanded")

    def __init__(self, prior: float):
        self.prior: float = prior
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.children: Dict[int, "Node"] = {}
        self.to_play: Optional[int] = None
        self.expanded: bool = False

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


def _terminal_value_for_side_to_move(game: FadingTicTacToe) -> float:
    """Ground-truth value of a terminal state, from the perspective of
    ``game.current_player`` (the side "to move" at this now-terminal
    node -- see the module docstring for why this is well-defined)."""
    winner = game.winner()
    if winner == 0 or winner is None:
        return 0.0
    return 1.0 if winner == game.current_player else -1.0


class MCTS:
    """Runs PUCT search rooted at a given game state, using a network
    for priors and leaf evaluation."""

    def __init__(self, network: PolicyValueNet, config: Config, device: torch.device,
                 rng: Optional[np.random.Generator] = None):
        self.network = network
        self.config = config
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    def run(self, root_game: FadingTicTacToe, add_dirichlet_noise: bool = False) -> Node:
        """Run ``config.num_simulations`` simulations from ``root_game``
        (which is not modified) and return the fully-searched root node."""
        root = Node(prior=1.0)
        root.to_play = root_game.current_player
        self._expand(root, root_game)

        if add_dirichlet_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            game = root_game.clone()
            node = root
            search_path: List[Node] = [node]

            while node.expanded and not game.is_terminal():
                action, node = self._select_child(node)
                game.make_move(action)
                search_path.append(node)

            if game.is_terminal():
                leaf_value = _terminal_value_for_side_to_move(game)
            else:
                node.to_play = game.current_player
                leaf_value = self._expand(node, game)

            self._backup(search_path, leaf_value)

        return root

    # ------------------------------------------------------------------
    def _select_child(self, node: Node) -> Tuple[int, Node]:
        """PUCT selection: argmax over children of
        Q + c_puct * P * sqrt(parent_visits) / (1 + child_visits).

        Critical perspective note: ``child.mean_value`` is stored from
        the perspective of the player to move *at the child* (see the
        module docstring) -- i.e. ``node``'s opponent. ``node`` must
        choose the action that is best for *itself*, so we negate the
        child's mean value to convert it into ``node``'s perspective
        before comparing. Forgetting this negation silently turns the
        search into one that actively seeks bad outcomes for the mover
        (verified against a random-agent baseline during development).
        """
        c_puct = self.config.c_puct
        sqrt_parent = math.sqrt(max(1, node.visit_count))

        best_score = -float("inf")
        best_action = -1
        best_child: Optional[Node] = None
        for action, child in node.children.items():
            q_from_node_perspective = -child.mean_value
            u = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q_from_node_perspective + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        assert best_child is not None, "A node marked expanded must have children."
        return best_action, best_child

    def _expand(self, node: Node, game: FadingTicTacToe) -> float:
        """Evaluate ``game`` with the network, mask illegal actions,
        populate ``node.children`` with priors, and return the value
        from the perspective of the player to move at ``game``."""
        state_tensor = game.to_tensor()
        priors, value = self.network.predict(state_tensor, self.device)

        legal = game.legal_moves()
        legal_mask = np.zeros(self.config.num_actions, dtype=np.float32)
        legal_mask[legal] = 1.0
        masked_priors = priors * legal_mask
        total = masked_priors.sum()
        if total > 1e-8:
            masked_priors = masked_priors / total
        else:
            # Degenerate case (network assigned ~0 prob to all legal
            # moves): fall back to a uniform prior over legal moves so
            # search can still proceed sensibly.
            masked_priors = legal_mask / max(1, len(legal))

        for action in legal:
            child = Node(prior=float(masked_priors[action]))
            node.children[action] = child
        node.expanded = True
        return value

    def _add_dirichlet_noise(self, root: Node) -> None:
        actions = list(root.children.keys())
        if not actions:
            return
        noise = self.rng.dirichlet([self.config.dirichlet_alpha] * len(actions))
        eps = self.config.dirichlet_epsilon
        for a, n in zip(actions, noise):
            child = root.children[a]
            child.prior = (1 - eps) * child.prior + eps * float(n)

    @staticmethod
    def _backup(search_path: List[Node], leaf_value: float) -> None:
        """Propagate ``leaf_value`` (from the perspective of the last
        node's ``to_play``) up through the path, flipping sign at every
        step since ``to_play`` strictly alternates level to level."""
        value = leaf_value
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += value
            value = -value

    # ------------------------------------------------------------------
    @staticmethod
    def visit_count_policy(root: Node, num_actions: int) -> np.ndarray:
        """Convert root visit counts into a normalized policy target
        over the fixed ``num_actions``-dimensional action space (zero
        everywhere except at legal actions that were expanded)."""
        counts = np.zeros(num_actions, dtype=np.float32)
        for action, child in root.children.items():
            counts[action] = child.visit_count
        total = counts.sum()
        if total > 0:
            counts = counts / total
        return counts

    @staticmethod
    def select_action(visit_policy: np.ndarray, temperature: float,
                       rng: np.random.Generator) -> int:
        """Sample (or greedily pick) an action from a visit-count policy.

        temperature == 0 (or very close to it) -> greedy argmax.
        temperature > 0 -> sample from counts^(1/temperature), renormalized.
        """
        if temperature <= 1e-3:
            return int(np.argmax(visit_policy))
        logits = np.log(np.clip(visit_policy, 1e-12, None)) / temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            # All-zero policy (shouldn't happen once root is expanded);
            # fall back to uniform over nonzero support of visit_policy.
            support = np.nonzero(visit_policy > 0)[0]
            if len(support) == 0:
                support = np.arange(len(visit_policy))
            return int(rng.choice(support))
        probs = probs / probs_sum
        return int(rng.choice(len(probs), p=probs))
