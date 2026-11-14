import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. TIC-TAC-TOE ENVIRONMENT
# ============================================================

class TicTacToe:

    def __init__(self):
        # 0 = empty
        # 1 = X
        # -1 = O
        self.board = np.zeros(9, dtype=np.int8)

        # Player whose turn it is
        self.player = 1

    def copy(self):
        game = TicTacToe()
        game.board = self.board.copy()
        game.player = self.player
        return game

    def legal_moves(self):
        return [
            i for i in range(9)
            if self.board[i] == 0
        ]

    def play(self, move):
        if self.board[move] != 0:
            raise ValueError("Illegal move")

        self.board[move] = self.player

        # Switch player
        self.player *= -1

    def winner(self):
        b = self.board.reshape(3, 3)

        lines = []

        # Rows
        lines.extend(b)

        # Columns
        lines.extend(b.T)

        # Diagonals
        lines.append(np.diag(b))
        lines.append(np.diag(np.fliplr(b)))

        for line in lines:

            s = np.sum(line)

            if s == 3:
                return 1

            if s == -3:
                return -1

        return 0

    def terminal(self):
        if self.winner() != 0:
            return True

        return len(self.legal_moves()) == 0

    def result(self):
        """
        Result from the perspective of the player
        who made the last move.
        """

        winner = self.winner()

        if winner == 0:
            return 0

        # The previous player made the winning move.
        previous_player = -self.player

        if winner == previous_player:
            return 1

        return -1

    def print_board(self):

        symbols = {
            1: "X",
            -1: "O",
            0: " "
        }

        print()

        for r in range(3):

            row = []

            for c in range(3):

                index = r * 3 + c

                value = self.board[index]

                if value == 0:
                    row.append(str(index + 1))
                else:
                    row.append(symbols[value])

            print(" " + " | ".join(row))

            if r < 2:
                print("---+---+---")

        print()


# ============================================================
# 2. ENCODING GAME STATE
# ============================================================

def encode(game):
    """
    Encode the board from the perspective
    of the current player.

    Current player's pieces  -> +1
    Opponent's pieces        -> -1
    Empty                    -> 0
    """

    board = game.board.astype(np.float32)

    # Change perspective
    board = board * game.player

    return torch.tensor(
        board,
        dtype=torch.float32
    )


# ============================================================
# 3. NEURAL NETWORK
# ============================================================

class AlphaZeroNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(9, 128)

        self.fc2 = nn.Linear(128, 128)

        # Policy head:
        # one output for each of the 9 moves
        self.policy_head = nn.Linear(128, 9)

        # Value head:
        # one number between -1 and +1
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        policy_logits = self.policy_head(x)

        value = self.value_head(x)

        value = torch.tanh(value)

        return policy_logits, value


# ============================================================
# 4. MCTS NODE
# ============================================================

class Node:

    def __init__(self, prior=0.0):

        # Neural network prior probability
        self.prior = float(prior)

        # Number of visits
        self.visit_count = 0

        # Sum of values
        self.value_sum = 0.0

        # action -> child Node
        self.children = {}

    @property
    def value(self):

        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count


# ============================================================
# 5. NEURAL NETWORK EVALUATION
# ============================================================

def evaluate(game, network, device):

    network.eval()

    state = encode(game)

    state = state.to(device)

    state = state.unsqueeze(0)

    with torch.no_grad():

        policy_logits, value = network(state)

    policy = torch.softmax(
        policy_logits,
        dim=1
    )[0]

    value = value.item()

    return policy.cpu().numpy(), value


# ============================================================
# 6. EXPAND NODE
# ============================================================

def expand(node, game, network, device):

    policy, value = evaluate(
        game,
        network,
        device
    )

    legal_moves = game.legal_moves()

    probabilities = np.zeros(9)

    for move in legal_moves:

        probabilities[move] = policy[move]

    # Mask illegal moves
    total = probabilities.sum()

    if total > 0:

        probabilities /= total

    else:

        # Safety fallback
        probability = 1.0 / len(legal_moves)

        for move in legal_moves:
            probabilities[move] = probability

    # Create child nodes
    for move in legal_moves:

        node.children[move] = Node(
            prior=probabilities[move]
        )

    return value


# ============================================================
# 7. PUCT SELECTION
# ============================================================

def select_child(node):

    best_score = -float("inf")

    best_action = None

    best_child = None

    # Exploration constant
    c_puct = 1.5

    sqrt_visits = math.sqrt(
        node.visit_count + 1
    )

    for action, child in node.children.items():

        # Exploitation
        q = child.value

        # Exploration
        u = (
            c_puct
            * child.prior
            * sqrt_visits
            / (1 + child.visit_count)
        )

        score = q + u

        if score > best_score:

            best_score = score

            best_action = action

            best_child = child

    return best_action, best_child


# ============================================================
# 8. BACKUP
# ============================================================

def backup(search_path, value):

    """
    Send the value back toward the root.

    Because this is a two-player zero-sum game,
    the value changes sign at every level.
    """

    for node in reversed(search_path):

        node.value_sum += value

        node.visit_count += 1

        # Switch perspective
        value = -value


# ============================================================
# 9. ONE MCTS SIMULATION
# ============================================================

def simulate(root, game, network, device):

    node = root

    current_game = game.copy()

    search_path = [node]

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    while node.children:

        action, node = select_child(node)

        current_game.play(action)

        search_path.append(node)

    # --------------------------------------------------------
    # Evaluation / Expansion
    # --------------------------------------------------------

    if current_game.terminal():

        winner = current_game.winner()

        if winner == 0:

            value = 0.0

        else:

            # From perspective of player whose turn it would be
            # after the last move.

            value = -1.0

    else:

        value = expand(
            node,
            current_game,
            network,
            device
        )

    # --------------------------------------------------------
    # Backup
    # --------------------------------------------------------

    backup(
        search_path,
        value
    )


# ============================================================
# 10. RUN MCTS
# ============================================================

def run_mcts(
    game,
    network,
    device,
    simulations=100
):

    root = Node(prior=1.0)

    # Expand root
    expand(
        root,
        game,
        network,
        device
    )

    # Run simulations
    for _ in range(simulations):

        simulate(
            root,
            game,
            network,
            device
        )

    # Convert visit counts into policy
    visits = np.zeros(9)

    for action, child in root.children.items():

        visits[action] = child.visit_count

    total = visits.sum()

    if total > 0:

        visits /= total

    return visits


# ============================================================
# 11. TEMPERATURE
# ============================================================

def apply_temperature(policy, temperature):

    if temperature <= 0:

        result = np.zeros_like(policy)

        result[np.argmax(policy)] = 1.0

        return result

    policy = np.power(
        policy,
        1.0 / temperature
    )

    total = policy.sum()

    if total == 0:

        legal = np.flatnonzero(policy)

        policy[legal] = 1.0 / len(legal)

        return policy

    return policy / total


# ============================================================
# 12. SELF PLAY
# ============================================================

def self_play(
    network,
    device,
    simulations=100
):

    game = TicTacToe()

    history = []

    while not game.terminal():

        # MCTS creates improved policy
        policy = run_mcts(
            game,
            network,
            device,
            simulations
        )

        # Temperature makes early game exploration stronger
        if len(history) < 5:

            temperature = 1.0

        else:

            temperature = 0.1

        policy = apply_temperature(
            policy,
            temperature
        )

        # Save state
        history.append(
            (
                game.board.copy(),
                game.player,
                policy.copy()
            )
        )

        # Choose move
        move = np.random.choice(
            9,
            p=policy
        )

        game.play(move)

    # --------------------------------------------------------
    # Game finished
    # --------------------------------------------------------

    winner = game.winner()

    data = []

    for board, player, policy in history:

        if winner == 0:

            z = 0.0

        elif winner == player:

            z = 1.0

        else:

            z = -1.0

        data.append(
            (
                board,
                player,
                policy,
                z
            )
        )

    return data


# ============================================================
# 13. TRAINING
# ============================================================

def train_step(
    network,
    optimizer,
    data,
    device
):

    network.train()

    random.shuffle(data)

    total_loss = 0.0

    for board, player, target_policy, target_value in data:

        # Perspective of current player
        board = board.astype(np.float32)

        board = board * player

        x = torch.tensor(
            board,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        target_policy = torch.tensor(
            target_policy,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        target_value = torch.tensor(
            [[target_value]],
            dtype=torch.float32,
            device=device
        )

        # Network prediction
        policy_logits, value = network(x)

        # ----------------------------------------------------
        # Policy loss
        # ----------------------------------------------------

        log_policy = F.log_softmax(
            policy_logits,
            dim=1
        )

        policy_loss = -(
            target_policy * log_policy
        ).sum()

        # ----------------------------------------------------
        # Value loss
        # ----------------------------------------------------

        value_loss = F.mse_loss(
            value,
            target_value
        )

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        loss = (
            policy_loss
            + value_loss
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(data), 1)


# ============================================================
# 14. TRAIN ALPHAZERO
# ============================================================

def train():

    # Select device
    if torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print("Device:", device)

    # Network
    network = AlphaZeroNet()

    network.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=0.001
    )

    # --------------------------------------------------------
    # Training parameters
    # --------------------------------------------------------

    iterations = 30

    games_per_iteration = 30

    simulations = 100

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for iteration in range(iterations):

        print()
        print(
            "=" * 60
        )

        print(
            f"Iteration {iteration + 1}/{iterations}"
        )

        training_data = []

        # ----------------------------------------------------
        # Self-play
        # ----------------------------------------------------

        for game_number in range(
            games_per_iteration
        ):

            data = self_play(
                network,
                device,
                simulations
            )

            training_data.extend(data)

            print(
                f"\rSelf-play: "
                f"{game_number + 1}/"
                f"{games_per_iteration}",
                end=""
            )

        print()

        print(
            "Training samples:",
            len(training_data)
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        loss = train_step(
            network,
            optimizer,
            training_data,
            device
        )

        print(
            f"Loss: {loss:.4f}"
        )

    # Save network
    torch.save(
        network.state_dict(),
        "alphazero_tictactoe.pt"
    )

    print()
    print(
        "Model saved as alphazero_tictactoe.pt"
    )

    return network, device


# ============================================================
# 15. AI VS HUMAN
# ============================================================

def play_against_ai(network, device):

    game = TicTacToe()

    print()
    print("Tic-Tac-Toe")
    print("You are X.")
    print("AI is O.")
    print()

    game.print_board()

    while not game.terminal():

        # ----------------------------------------------------
        # Human
        # ----------------------------------------------------

        if game.player == 1:

            while True:

                try:

                    move = int(
                        input(
                            "Choose a position (1-9): "
                        )
                    ) - 1

                    if move in game.legal_moves():
                        break

                    print(
                        "Illegal move."
                    )

                except ValueError:

                    print(
                        "Please enter a number."
                    )

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        else:

            print(
                "AI is thinking..."
            )

            policy = run_mcts(
                game,
                network,
                device,
                simulations=500
            )

            move = int(
                np.argmax(policy)
            )

            print(
                "AI chooses:",
                move + 1
            )

        game.play(move)

        game.print_board()

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    winner = game.winner()

    if winner == 1:

        print("You won!")

    elif winner == -1:

        print("AI won!")

    else:

        print("Draw!")


# ============================================================
# 16. MAIN
# ============================================================

if __name__ == "__main__":

    network, device = train()

    play_against_ai(
        network,
        device
    )