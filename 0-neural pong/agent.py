import random

import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from replay_buffer import ReplayBuffer


class Agent:

    def __init__(self):

        # -------------------------
        # Networks
        # -------------------------

        self.model = DQN()

        self.target_model = DQN()

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.target_model.eval()

        # -------------------------
        # Optimizer
        # -------------------------

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        self.loss_function = nn.MSELoss()

        # -------------------------
        # Replay Buffer
        # -------------------------

        self.memory = ReplayBuffer(
            capacity=50000
        )

        # -------------------------
        # DQN parameters
        # -------------------------

        self.gamma = 0.99

        self.batch_size = 64

        # -------------------------
        # Exploration
        # -------------------------

        self.epsilon = 1.0

        self.epsilon_min = 0.05

        self.epsilon_decay = 0.995

        # -------------------------
        # Target Network
        # -------------------------

        self.target_update_frequency = 1000

        self.training_steps = 0

    # ==================================================
    # Choose Action
    # ==================================================

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:

            return random.randint(0, 2)

        # Exploitation
        state = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.model(state)

        return torch.argmax(
            q_values,
            dim=1
        ).item()

    # ==================================================
    # Save Experience
    # ==================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    # ==================================================
    # Train
    # ==================================================

    def train_step(self):

        # هنوز تجربه کافی نداریم
        if len(self.memory) < self.batch_size:
            return None

        # -------------------------
        # Sample batch
        # -------------------------

        batch = self.memory.sample(
            self.batch_size
        )

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for (
            state,
            action,
            reward,
            next_state,
            done
        ) in batch:

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # -------------------------
        # Convert to tensors
        # -------------------------

        states = torch.tensor(
            states,
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )

        # -------------------------
        # Current Q
        # -------------------------

        q_values = self.model(states)

        current_q = q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # -------------------------
        # Target Q
        # -------------------------

        with torch.no_grad():

            next_q_values = self.target_model(
                next_states
            )

            max_next_q = next_q_values.max(
                dim=1
            ).values

            target_q = rewards + (
                1 - dones
            ) * self.gamma * max_next_q

        # -------------------------
        # Loss
        # -------------------------

        loss = self.loss_function(
            current_q,
            target_q
        )

        # -------------------------
        # Backpropagation
        # -------------------------

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # -------------------------
        # Target Network update
        # -------------------------

        self.training_steps += 1

        if (
            self.training_steps
            % self.target_update_frequency
            == 0
        ):

            self.target_model.load_state_dict(
                self.model.state_dict()
            )

        return loss.item()

    # ==================================================
    # Epsilon Decay
    # ==================================================

    def update_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )