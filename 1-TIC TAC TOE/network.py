import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)

        # Policy head
        self.policy_head = nn.Linear(128, 9)

        # Value head
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))

        return policy_logits, value