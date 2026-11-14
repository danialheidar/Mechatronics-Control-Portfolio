import torch

from network.model import AlphaZeroNetwork
from mcts.mcts import MCTS
from self_play.self_play import SelfPlay
from training.replay_buffer import ReplayBuffer
from training.trainer import Trainer


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    network = AlphaZeroNetwork(
        input_size=56,
        action_size=150
    ).to(DEVICE)

    checkpoint_path = "checkpoint_2050.pt"

    network.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)
    )

    mcts = MCTS(
        network=network,
        simulations=100,
        c_puct=1.5,
        device=DEVICE
    )

    self_play = SelfPlay(
        mcts=mcts,
        temperature=1.0
    )

    replay_buffer = ReplayBuffer(
        capacity=100000
    )

    trainer = Trainer(
        network=network,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device=DEVICE
    )

    for iteration in range(2051, 5000):

        print(
            f"\nIteration {iteration}"
        )

        # =============================================
        # Self Play
        # =============================================

        examples = self_play.play_game()

        for state, policy, value in examples:

            replay_buffer.add(
                state,
                policy,
                value
            )

        print(
            "Replay buffer:",
            len(replay_buffer)
        )

        # =============================================
        # Training
        # =============================================

        if len(replay_buffer) < 1024:
            continue

        for _ in range(100):

            states, policies, values = (
                replay_buffer.sample(256)
            )

            metrics = trainer.train_batch(
                states,
                policies,
                values
            )

        print(metrics)

        # =============================================
        # Checkpoint
        # =============================================

        if iteration % 10 == 0:

            torch.save(
                network.state_dict(),
                f"checkpoint_{iteration}.pt"
            )


if __name__ == "__main__":
    main()