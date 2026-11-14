import sys

import pygame
import torch

from game import TicTacToe
from network import AlphaZeroNet
from mcts import run_mcts


# =========================
# Configuration
# =========================

WINDOW_SIZE = 600
BOARD_SIZE = 600

LINE_WIDTH = 6

FPS = 60

AI_SIMULATIONS = 300

BACKGROUND = (245, 245, 245)
LINE_COLOR = (40, 40, 40)

X_COLOR = (40, 100, 200)
O_COLOR = (220, 70, 70)

TEXT_COLOR = (40, 40, 40)

CELL_SIZE = BOARD_SIZE // 3


# =========================
# Model
# =========================

def load_model(path="alphazero_tictactoe.pth"):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    network = AlphaZeroNet().to(device)

    network.load_state_dict(
        torch.load(
            path,
            map_location=device
        )
    )

    network.eval()

    return network, device


# =========================
# Drawing
# =========================

def draw_board(screen):
    screen.fill(BACKGROUND)

    # Vertical lines
    pygame.draw.line(
        screen,
        LINE_COLOR,
        (CELL_SIZE, 0),
        (CELL_SIZE, BOARD_SIZE),
        LINE_WIDTH
    )

    pygame.draw.line(
        screen,
        LINE_COLOR,
        (CELL_SIZE * 2, 0),
        (CELL_SIZE * 2, BOARD_SIZE),
        LINE_WIDTH
    )

    # Horizontal lines
    pygame.draw.line(
        screen,
        LINE_COLOR,
        (0, CELL_SIZE),
        (BOARD_SIZE, CELL_SIZE),
        LINE_WIDTH
    )

    pygame.draw.line(
        screen,
        LINE_COLOR,
        (0, CELL_SIZE * 2),
        (BOARD_SIZE, CELL_SIZE * 2),
        LINE_WIDTH
    )


def draw_x(screen, row, col):
    padding = 45

    x1 = col * CELL_SIZE + padding
    y1 = row * CELL_SIZE + padding

    x2 = (col + 1) * CELL_SIZE - padding
    y2 = (row + 1) * CELL_SIZE - padding

    pygame.draw.line(
        screen,
        X_COLOR,
        (x1, y1),
        (x2, y2),
        12
    )

    pygame.draw.line(
        screen,
        X_COLOR,
        (x2, y1),
        (x1, y2),
        12
    )


def draw_o(screen, row, col):
    center_x = col * CELL_SIZE + CELL_SIZE // 2
    center_y = row * CELL_SIZE + CELL_SIZE // 2

    radius = CELL_SIZE // 2 - 45

    pygame.draw.circle(
        screen,
        O_COLOR,
        (center_x, center_y),
        radius,
        12
    )


def draw_pieces(screen, game):
    for index, value in enumerate(game.board):

        row = index // 3
        col = index % 3

        if value == 1:
            draw_x(screen, row, col)

        elif value == -1:
            draw_o(screen, row, col)


def draw_text(screen, text, font, y):
    surface = font.render(
        text,
        True,
        TEXT_COLOR
    )

    rect = surface.get_rect(
        center=(WINDOW_SIZE // 2, y)
    )

    screen.blit(surface, rect)


# =========================
# Game Over Screen
# =========================

def draw_game_over(screen, game):
    overlay = pygame.Surface(
        (WINDOW_SIZE, WINDOW_SIZE),
        pygame.SRCALPHA
    )

    overlay.fill(
        (255, 255, 255, 190)
    )

    screen.blit(overlay, (0, 0))

    font = pygame.font.Font(
        None,
        64
    )

    small_font = pygame.font.Font(
        None,
        32
    )

    result = game.result()

    if result == 1:
        message = "You Win!"

    elif result == -1:
        message = "AI Wins!"

    else:
        message = "Draw!"

    draw_text(
        screen,
        message,
        font,
        WINDOW_SIZE // 2 - 30
    )

    draw_text(
        screen,
        "Press R to play again",
        small_font,
        WINDOW_SIZE // 2 + 40
    )

    draw_text(
        screen,
        "Press ESC to quit",
        small_font,
        WINDOW_SIZE // 2 + 80
    )


# =========================
# Mouse
# =========================

def mouse_to_action(position):
    x, y = position

    col = x // CELL_SIZE
    row = y // CELL_SIZE

    if 0 <= row < 3 and 0 <= col < 3:
        return row * 3 + col

    return None


# =========================
# AI Move
# =========================

def get_ai_move(game, network, device):
    print("AI is thinking...")

    visits = run_mcts(
        game,
        network,
        simulations=AI_SIMULATIONS,
        device=device
    )

    action = int(visits.argmax())

    print(
        f"AI chooses cell {action}"
    )

    return action


# =========================
# Main Game
# =========================

def main():
    pygame.init()

    screen = pygame.display.set_mode(
        (WINDOW_SIZE, WINDOW_SIZE)
    )

    pygame.display.set_caption(
        "AlphaZero Tic-Tac-Toe"
    )

    clock = pygame.time.Clock()

    network, device = load_model()

    font = pygame.font.Font(
        None,
        36
    )

    game = TicTacToe()

    running = True
    ai_thinking = False

    while running:

        clock.tick(FPS)

        for event in pygame.event.get():

            # Close window
            if event.type == pygame.QUIT:
                running = False

            # Keyboard
            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif (
                    event.key == pygame.K_r
                    and game.is_terminal()
                ):
                    game = TicTacToe()
                    ai_thinking = False

            # Mouse
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if (
                    event.button == 1
                    and not game.is_terminal()
                    and game.player == 1
                    and not ai_thinking
                ):
                    action = mouse_to_action(
                        event.pos
                    )

                    if (
                        action is not None
                        and action in game.legal_actions()
                    ):
                        game.play(action)

        # =====================
        # AI turn
        # =====================

        if (
            not game.is_terminal()
            and game.player == -1
            and not ai_thinking
        ):
            ai_thinking = True

            # Draw current board before AI thinking
            draw_board(screen)
            draw_pieces(screen, game)

            draw_text(
                screen,
                "AI is thinking...",
                font,
                30
            )

            pygame.display.flip()

            # Calculate AI move
            action = get_ai_move(
                game,
                network,
                device
            )

            game.play(action)

            ai_thinking = False

        # =====================
        # Draw
        # =====================

        draw_board(screen)
        draw_pieces(screen, game)

        if not game.is_terminal():

            if game.player == 1:
                status = "Your turn - X"

            else:
                status = "AI turn - O"

            draw_text(
                screen,
                status,
                font,
                30
            )

        else:
            draw_game_over(
                screen,
                game
            )

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()