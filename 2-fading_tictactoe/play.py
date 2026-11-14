"""
play.py
=======

Command-line interface for a human to play Fading-Move Tic-Tac-Toe
against a trained AlphaZero agent.

This uses the exact same ``FadingTicTacToe`` engine (game.py) as
training and evaluation, so there is no risk of the human-play rules
drifting from the rules the agent was trained under.

Usage:
    python play.py --checkpoint checkpoints/best.pt
    python play.py --checkpoint checkpoints/best.pt --human-plays O
    python play.py --checkpoint checkpoints/best.pt --simulations 400
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from config import Config
from evaluate import NetworkAgent
from game import FadingTicTacToe, PLAYER_X, PLAYER_O
from network import PolicyValueNet


_SYMBOL = {PLAYER_X: "X", PLAYER_O: "O", 0: "."}


def render_board(game: FadingTicTacToe) -> str:
    rows = []
    for r in range(3):
        cells = []
        for c in range(3):
            idx = r * 3 + c
            cells.append(_SYMBOL[game.board[idx]])
        rows.append(" ".join(cells))
    return "\n".join(rows)


def describe_fade_warning(game: FadingTicTacToe) -> str:
    """Human-readable note about whose mark will fade on their next move."""
    lines = []
    for player, name in ((PLAYER_X, "X"), (PLAYER_O, "O")):
        if len(game.history[player]) >= game.max_active_marks:
            oldest = game.history[player][0]
            lines.append(f"  ({name}'s next move will fade square {oldest})")
    return "\n".join(lines)


def print_board(game: FadingTicTacToe, last_faded: int | None = None) -> None:
    print()
    print(render_board(game))
    warning = describe_fade_warning(game)
    if warning:
        print(warning)
    if last_faded is not None:
        print(f"  (square {last_faded} just faded)")
    print()


def get_human_move(game: FadingTicTacToe) -> int:
    legal = set(game.legal_moves())
    while True:
        raw = input(f"Your move {sorted(legal)}: ").strip()
        try:
            move = int(raw)
        except ValueError:
            print("Please enter a number 0-8.")
            continue
        if move not in legal:
            print(f"Square {move} is not a legal move right now.")
            continue
        return move


def load_agent(checkpoint_path: str | None, device: torch.device, num_simulations: int) -> NetworkAgent:
    if checkpoint_path is None:
        print("WARNING: no --checkpoint given; using an untrained (random-weight) "
              "network. Pass --checkpoint checkpoints/best.pt for a real opponent.")
        config = Config()
        network = PolicyValueNet(config).to(device)
        network.eval()
        return NetworkAgent(network, config, device, num_simulations=num_simulations, temperature=0.0)

    from train import load_checkpoint

    network, _optimizer, _iteration, config = load_checkpoint(checkpoint_path, device)
    network.eval()
    return NetworkAgent(network, config, device, num_simulations=num_simulations, temperature=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against a trained Fading-Move Tic-Tac-Toe agent.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained checkpoint.")
    parser.add_argument("--human-plays", type=str, default="X", choices=["X", "O"],
                         help="Which side the human plays.")
    parser.add_argument("--simulations", type=int, default=400,
                         help="MCTS simulations per move for the agent.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    agent = load_agent(args.checkpoint, device, args.simulations)
    game = FadingTicTacToe(agent.config)

    human_player = PLAYER_X if args.human_plays == "X" else PLAYER_O

    print("Fading-Move Tic-Tac-Toe")
    print("Squares are numbered:")
    print("0 1 2\n3 4 5\n6 7 8")
    print(f"You are {'X' if human_player == PLAYER_X else 'O'}.")
    print_board(game)

    while not game.is_terminal():
        if game.current_player == human_player:
            move = get_human_move(game)
        else:
            print("Agent is thinking...")
            move = agent.choose_action(game, rng)
            print(f"Agent plays {move}.")
        game.make_move(move)
        print_board(game, last_faded=game.last_faded_square)

    winner = game.winner()
    if winner == 0:
        print("Result: draw.")
    elif winner == human_player:
        print("Result: you win!")
    else:
        print("Result: the agent wins.")


if __name__ == "__main__":
    main()
