"""
evaluate.py
===========

Head-to-head evaluation between agents:

* ``RandomAgent``   -- uniformly picks among legal moves.
* ``NetworkAgent``  -- the AlphaZero agent: network-guided PUCT MCTS.
  This is the only agent whose *decision making* is used to judge
  training progress; it never consults the minimax agent.
* ``MinimaxAgent``  -- a depth-limited, alpha-beta-pruned reference
  search agent with a hand-written heuristic, used only as an
  independent sanity-check opponent. Because Fading-Move Tic-Tac-Toe
  can cycle indefinitely, this is deliberately *not* a perfect solver:
  it searches to a fixed ply depth and falls back to a static
  line-based heuristic at the frontier. Treat it as "a reasonably
  strong fixed baseline", not ground truth.

``evaluate_networks`` (or more generally ``evaluate_agents``) plays a
head-to-head match between two agents, alternating which one plays X
vs O each game for fairness, and reports a simple win/loss/draw
tally from the perspective of the first agent passed in.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import torch

from config import Config
from game import FadingTicTacToe, LINES, PLAYER_X, PLAYER_O
from mcts import MCTS, _terminal_value_for_side_to_move
from network import PolicyValueNet


class Agent(Protocol):
    def choose_action(self, game: FadingTicTacToe, rng: np.random.Generator) -> int:
        ...


# ----------------------------------------------------------------------
# Random agent
# ----------------------------------------------------------------------
class RandomAgent:
    def choose_action(self, game: FadingTicTacToe, rng: np.random.Generator) -> int:
        moves = game.legal_moves()
        return int(rng.choice(moves))


# ----------------------------------------------------------------------
# Network (AlphaZero) agent
# ----------------------------------------------------------------------
class NetworkAgent:
    """Wraps a trained network behind greedy (or low-temperature) PUCT
    MCTS action selection -- this is the actual AlphaZero player."""

    def __init__(
        self,
        network: PolicyValueNet,
        config: Config,
        device: torch.device,
        num_simulations: Optional[int] = None,
        temperature: float = 0.0,
    ):
        self.network = network
        self.device = device
        self.temperature = temperature
        # Use a private copy of the config so we can override the
        # simulation budget for evaluation without mutating the caller's
        # training configuration.
        self.config = copy.deepcopy(config)
        if num_simulations is not None:
            self.config.num_simulations = num_simulations

    def choose_action(self, game: FadingTicTacToe, rng: np.random.Generator) -> int:
        mcts = MCTS(self.network, self.config, self.device, rng)
        root = mcts.run(game, add_dirichlet_noise=True)
        visit_policy = MCTS.visit_count_policy(root, self.config.num_actions)
        return MCTS.select_action(visit_policy, self.temperature, rng)


# ----------------------------------------------------------------------
# Minimax reference agent (approximate -- see module docstring)
# ----------------------------------------------------------------------
def _heuristic_value_for_side_to_move(game: FadingTicTacToe) -> float:
    """A cheap, hand-written static evaluation, returned from the
    perspective of ``game.current_player``. Positive favors X overall;
    the final sign flip maps that into "goodness for whoever is to
    move". This is only used at the minimax search frontier."""
    active_x = set(game.history[PLAYER_X])
    active_o = set(game.history[PLAYER_O])
    score = 0.0
    for line in LINES:
        xs = sum(1 for i in line if i in active_x)
        os_ = sum(1 for i in line if i in active_o)
        if xs > 0 and os_ > 0:
            continue  # blocked line: contributes nothing to either side
        if xs > 0:
            score += xs ** 2
        elif os_ > 0:
            score -= os_ ** 2
    normalized = float(np.tanh(score / 6.0))
    return game.current_player * normalized


def _negamax(game: FadingTicTacToe, depth: int, alpha: float, beta: float) -> float:
    """Negamax with alpha-beta pruning. Returns a value from the
    perspective of ``game.current_player``."""
    if game.is_terminal():
        return _terminal_value_for_side_to_move(game)
    if depth == 0:
        return _heuristic_value_for_side_to_move(game)

    best = -float("inf")
    for action in game.legal_moves():
        child = game.clone()
        child.make_move(action)
        value = -_negamax(child, depth - 1, -beta, -alpha)
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best


class MinimaxAgent:
    """Depth-limited alpha-beta search with a static heuristic frontier.

    NOT a perfect solver (the game's cyclic nature and branching factor
    make exhaustive solving impractical to guarantee termination cleanly
    within this project's scope); intended purely as an independent
    fixed-strength reference opponent for sanity-checking the AlphaZero
    agent's progress.
    """

    def __init__(self, depth: int = 5):
        self.depth = depth

    def choose_action(self, game: FadingTicTacToe, rng: np.random.Generator) -> int:
        best_value = -float("inf")
        best_actions = []
        for action in game.legal_moves():
            child = game.clone()
            child.make_move(action)
            value = -_negamax(child, self.depth - 1, -float("inf"), float("inf"))
            if value > best_value + 1e-9:
                best_value = value
                best_actions = [action]
            elif abs(value - best_value) <= 1e-9:
                best_actions.append(action)
        return int(rng.choice(best_actions))


# ----------------------------------------------------------------------
# Match running
# ----------------------------------------------------------------------
@dataclass
class EvalResult:
    wins: int
    losses: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games > 0 else 0.0


def play_game(
    agent_for_x: Agent, agent_for_o: Agent, config: Config, rng: np.random.Generator
) -> Optional[int]:
    game = FadingTicTacToe(config)
    while not game.is_terminal():
        agent = agent_for_x if game.current_player == PLAYER_X else agent_for_o
        action = agent.choose_action(game, rng)
        game.make_move(action)
    return game.winner()


def evaluate_agents(
    agent_a: Agent,
    agent_b: Agent,
    num_games: int,
    rng: np.random.Generator,
    config: Optional[Config] = None,
) -> EvalResult:
    """Play ``num_games`` games between ``agent_a`` and ``agent_b``,
    alternating who plays X each game, and tally the result from
    ``agent_a``'s perspective."""
    cfg = config or Config()
    wins = losses = draws = 0
    for i in range(num_games):
        a_plays_x = (i % 2 == 0)
        x_agent, o_agent = (agent_a, agent_b) if a_plays_x else (agent_b, agent_a)
        winner = play_game(x_agent, o_agent, cfg, rng)
        if winner == 0 or winner is None:
            draws += 1
        elif (winner == PLAYER_X) == a_plays_x:
            wins += 1
        else:
            losses += 1
    return EvalResult(wins=wins, losses=losses, draws=draws)


# Backwards/semantic-clarity alias used by train.py.
evaluate_networks = evaluate_agents


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _load_network_agent(checkpoint_path: str, device: torch.device, num_simulations: int) -> NetworkAgent:
    # Imported here to avoid a circular import (train.py imports evaluate.py).
    from train import load_checkpoint

    network, _optimizer, _iteration, config = load_checkpoint(checkpoint_path, device)
    network.eval()
    return NetworkAgent(network, config, device, num_simulations=num_simulations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained AlphaZero checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a network checkpoint.")
    parser.add_argument("--opponent", type=str, default="random",
                         choices=["random", "minimax", "checkpoint"],
                         help="Opponent type.")
    parser.add_argument("--opponent-checkpoint", type=str, default=None,
                         help="Path to opponent checkpoint, if --opponent=checkpoint.")
    parser.add_argument("--games", type=int, default=40, help="Number of games to play.")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move for network agent(s).")
    parser.add_argument("--minimax-depth", type=int, default=5, help="Search depth for the minimax agent.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    agent_a = _load_network_agent(args.checkpoint, device, args.simulations)

    if args.opponent == "random":
        agent_b: Agent = RandomAgent()
    elif args.opponent == "minimax":
        agent_b = MinimaxAgent(depth=args.minimax_depth)
    else:
        if not args.opponent_checkpoint:
            raise ValueError("--opponent-checkpoint is required when --opponent=checkpoint")
        agent_b = _load_network_agent(args.opponent_checkpoint, device, args.simulations)

    result = evaluate_agents(agent_a, agent_b, args.games, rng, config=agent_a.config)
    print(f"Results over {result.games} games vs {args.opponent}:")
    print(f"  Wins:   {result.wins}")
    print(f"  Losses: {result.losses}")
    print(f"  Draws:  {result.draws}")
    print(f"  Win rate (of decisive games): "
          f"{(result.wins / (result.wins + result.losses)) if (result.wins + result.losses) > 0 else float('nan'):.3f}")
    print(f"  Win rate (of all games): {result.win_rate:.3f}")


if __name__ == "__main__":
    main()
