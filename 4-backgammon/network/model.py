import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, size):

        super().__init__()

        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)

        self.relu = nn.ReLU()

    def forward(self, x):

        residual = x

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        x = x + residual
        x = self.relu(x)

        return x


class AlphaZeroNetwork(nn.Module):

    def __init__(
        self,
        input_size=56,
        action_size=150,
        hidden_size=256
    ):

        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        self.blocks = nn.Sequential(
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size)
        )

        # Policy

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

        # Value

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.input_layer(x)

        x = self.blocks(x)

        policy = self.policy_head(x)

        value = self.value_head(x)

        return policy, value