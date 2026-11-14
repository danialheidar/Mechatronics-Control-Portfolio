import numpy as np

from game import TicTacToe
from mcts import run_mcts


def visit_distribution(visits, temperature=1.0):
    visits = np.asarray(visits, dtype=np.float32)

    if temperature <= 0:
        policy = np.zeros_like(visits)
        policy[np.argmax(visits)] = 1.0
        return policy

    visits = visits ** (1.0 / temperature)

    total = visits.sum()

    if total == 0:
        legal = np.where(visits >= 0)[0]
        policy = np.zeros_like(visits)
        policy[legal] = 1.0 / len(legal)
        return policy

    return visits / total


def self_play(
    network,
    simulations=100,
    temperature=1.0,
    device="cpu"
):
    game = TicTacToe()

    history = []

    while not game.is_terminal():

        state = game.encode().copy()

        visits = run_mcts(
            game,
            network,
            simulations=simulations,
            device=device
        )

        policy = visit_distribution(
            visits,
            temperature
        )

        history.append(
            (
                state,
                policy,
                game.player
            )
        )

        action = np.random.choice(
            9,
            p=policy
        )

        game.play(action)

    result = game.result()

    training_data = []

    for state, policy, player in history:

        # result را از دید بازیکنی که state را دیده تبدیل می‌کنیم
        value = result * player

        training_data.append(
            (
                state,
                policy,
                value
            )
        )

    return training_data