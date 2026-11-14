import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.player = 1

    def clone(self):
        game = TicTacToe()
        game.board = self.board.copy()
        game.player = self.player
        return game

    def legal_actions(self):
        return np.where(self.board == 0)[0].tolist()

    def play(self, action):
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")

        self.board[action] = self.player
        self.player *= -1

    def winner(self):
        winning_lines = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]

        for a, b, c in winning_lines:
            if (
                self.board[a] != 0
                and self.board[a] == self.board[b]
                and self.board[b] == self.board[c]
            ):
                return int(self.board[a])

        return 0

    def is_draw(self):
        return self.winner() == 0 and len(self.legal_actions()) == 0

    def is_terminal(self):
        return self.winner() != 0 or self.is_draw()

    def result(self):
        winner = self.winner()

        if winner == 1:
            return 1

        if winner == -1:
            return -1

        return 0

    def encode(self):
        """
        Board را از دید بازیکن فعلی encode می‌کند.

        خانه‌های بازیکن فعلی: +1
        خانه‌های حریف: -1
        خانه خالی: 0
        """

        return self.board * self.player

    def display(self):
        symbols = {
            1: "X",
            -1: "O",
            0: " "
        }

        print()

        for row in range(3):
            cells = []

            for col in range(3):
                index = row * 3 + col
                cells.append(symbols[int(self.board[index])])

            print(f" {cells[0]} | {cells[1]} | {cells[2]}")

            if row < 2:
                print("---+---+---")

        print()