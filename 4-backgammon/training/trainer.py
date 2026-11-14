import torch
import torch.nn.functional as F


class Trainer:

    def __init__(
        self,
        network,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device="cpu"
    ):

        self.network = network
        self.device = device

        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train_batch(
        self,
        states,
        target_policies,
        target_values
    ):

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        target_policies = torch.tensor(
            target_policies,
            dtype=torch.float32,
            device=self.device
        )

        target_values = torch.tensor(
            target_values,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        policy_logits, values = self.network(
            states
        )

        log_probs = F.log_softmax(
            policy_logits,
            dim=1
        )

        policy_loss = -(
            target_policies * log_probs
        ).sum(dim=1).mean()

        value_loss = F.mse_loss(
            values,
            target_values
        )

        loss = (
            policy_loss
            + value_loss
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            1.0
        )

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }