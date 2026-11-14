import math
import numpy as np
import torch

from game import TicTacToe


class Node:
    def __init__(self, game, parent=None, prior=0.0, action=None):
        self.game = game
        self.parent = parent
        self.prior = prior
        self.action = action

        self.visit_count = 0
        self.value_sum = 0.0

        self.children = {}

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0


def evaluate_network(network, game, device="cpu"):
    state = torch.tensor(
        game.encode(),
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        policy_logits, value = network(state)

    policy = torch.softmax(policy_logits, dim=1)[0]
    value = value.item()

    return policy.cpu().numpy(), value


def expand_node(node, network, device="cpu"):
    policy, value = evaluate_network(
        network,
        node.game,
        device
    )

    legal_actions = node.game.legal_actions()

    # فقط حرکات قانونی
    legal_policy = np.zeros(9, dtype=np.float32)

    for action in legal_actions:
        legal_policy[action] = policy[action]

    # Normalize
    total = legal_policy.sum()

    if total > 0:
        legal_policy /= total
    else:
        legal_policy[legal_actions] = 1.0 / len(legal_actions)

    for action in legal_actions:
        child_game = node.game.clone()
        child_game.play(action)

        child = Node(
            game=child_game,
            parent=node,
            prior=float(legal_policy[action]),
            action=action
        )

        node.children[action] = child

    return value


def select_child(node, c_puct=1.5):
    total_visits = sum(
        child.visit_count
        for child in node.children.values()
    )

    best_score = -float("inf")
    best_child = None

    for child in node.children.values():

        # چون value مربوط به دید بازیکن داخل child است،
        # برای parent باید علامت آن برعکس شود.
        q = -child.value

        u = (
            c_puct
            * child.prior
            * math.sqrt(total_visits + 1)
            / (1 + child.visit_count)
        )

        score = q + u

        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def backup(node, value):
    while node is not None:
        node.visit_count += 1
        node.value_sum += value

        # تغییر perspective
        value = -value

        node = node.parent


def simulate(root, network, device="cpu", c_puct=1.5):
    node = root

    # Selection
    while node.is_expanded() and not node.game.is_terminal():
        node = select_child(node, c_puct)

    # Terminal state
    if node.game.is_terminal():

        result = node.game.result()

        if result == 0:
            value = 0.0
        else:
            # player فعلی، بازیکنی نیست که حرکت برنده را انجام داده
            value = -1.0

        backup(node, value)
        return

    # Expansion + Evaluation
    value = expand_node(
        node,
        network,
        device
    )

    backup(node, value)


def run_mcts(
    game,
    network,
    simulations=100,
    device="cpu",
    c_puct=1.5
):
    root = Node(game.clone())

    # Expand root
    expand_node(
        root,
        network,
        device
    )

    for _ in range(simulations):
        simulate(
            root,
            network,
            device,
            c_puct
        )

    visits = np.zeros(9, dtype=np.float32)

    for action, child in root.children.items():
        visits[action] = child.visit_count

    return visits