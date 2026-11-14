from collections import deque
import random


class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def sample(self, batch_size):
        return random.sample(
            self.buffer,
            batch_size
        )

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":

    memory = ReplayBuffer(1000)

    memory.push(
        [0.5, 0.5, 1, -1, 0.5],
        1,
        0,
        [0.51, 0.48, 1, -1, 0.5],
        False
    )

    print("Buffer size:", len(memory))

    batch = memory.sample(1)

    print("Sample:")
    print(batch)