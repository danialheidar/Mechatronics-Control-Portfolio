"""
play_gui.py
===========

Pygame GUI for playing Fading-Move Tic-Tac-Toe against a trained
AlphaZero agent.

Uses the exact same FadingTicTacToe engine and NetworkAgent as
training/evaluation.

Usage:
    python play_gui.py --checkpoint checkpoints/best.pt
    python play_gui.py --checkpoint checkpoints/best.pt --human-plays O
    python play_gui.py --checkpoint checkpoints/best.pt --simulations 400
    python play_gui.py --checkpoint checkpoints/best.pt --seed 42

Install:
    pip install pygame-ce
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pygame
import torch

from config import Config
from evaluate import NetworkAgent
from game import FadingTicTacToe, PLAYER_X, PLAYER_O
from network import PolicyValueNet


# ================================================================
# Window configuration
# ================================================================

WINDOW_WIDTH = 700
WINDOW_HEIGHT = 830

BOARD_SIZE = 600
CELL_SIZE = BOARD_SIZE // 3

BOARD_X = (WINDOW_WIDTH - BOARD_SIZE) // 2
BOARD_Y = 120

FPS = 60


# ================================================================
# Colors
# ================================================================

BACKGROUND = (245, 245, 240)

BOARD_BACKGROUND = (252, 252, 248)

GRID_COLOR = (55, 55, 55)

TEXT_COLOR = (35, 35, 35)

X_COLOR = (40, 90, 180)
O_COLOR = (190, 60, 60)

HOVER_COLOR = (230, 230, 220)

FADE_COLOR = (220, 170, 50)

BUTTON_COLOR = (65, 65, 65)
BUTTON_HOVER_COLOR = (85, 85, 85)
BUTTON_TEXT_COLOR = (255, 255, 255)

WIN_COLOR = (40, 150, 70)

INVALID_COLOR = (180, 60, 60)


# ================================================================
# Font handling
# ================================================================

def load_font(size: int) -> pygame.font.Font:
    """
    Load a font without using pygame.font.SysFont().

    SysFont() can cause an UnboundLocalError in some pygame-ce /
    Python / Windows combinations, so we explicitly look for
    installed font files.
    """

    font_paths = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\verdana.ttf",
    ]

    for path in font_paths:
        if os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except pygame.error:
                pass

    # Final fallback.
    return pygame.font.Font(None, size)


def load_fonts() -> dict[str, pygame.font.Font]:
    """Load all fonts used by the GUI."""

    return {
        "title": load_font(42),
        "status": load_font(36),
        "info": load_font(25),
        "square": load_font(26),
        "fade": load_font(24),
        "button": load_font(26),
        "result": load_font(48),
    }


# ================================================================
# Agent
# ================================================================

def load_agent(
    checkpoint_path: str | None,
    device: torch.device,
    num_simulations: int,
) -> NetworkAgent:
    """
    Load the trained AlphaZero network.

    If no checkpoint is provided, a randomly initialized network
    is used.
    """

    if checkpoint_path is None:
        print(
            "WARNING: no --checkpoint given; using an untrained "
            "(random-weight) network."
        )

        config = Config()

        network = PolicyValueNet(config).to(device)
        network.eval()

        return NetworkAgent(
            network,
            config,
            device,
            num_simulations=num_simulations,
            temperature=0.0,
        )

    print(f"Loading checkpoint: {checkpoint_path}")

    from train import load_checkpoint

    network, _optimizer, _iteration, config = load_checkpoint(
        checkpoint_path,
        device,
    )

    network.eval()

    print("Checkpoint loaded successfully.")

    return NetworkAgent(
        network,
        config,
        device,
        num_simulations=num_simulations,
        temperature=0.0,
    )


# ================================================================
# Drawing utilities
# ================================================================

def draw_text(
    surface: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
    center: tuple[int, int],
) -> None:
    """Draw centered text."""

    rendered = font.render(text, True, color)

    rect = rendered.get_rect(
        center=center
    )

    surface.blit(rendered, rect)


def draw_x(
    screen: pygame.Surface,
    center_x: int,
    center_y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw an X mark."""

    padding = 45

    pygame.draw.line(
        screen,
        color,
        (
            center_x - padding,
            center_y - padding,
        ),
        (
            center_x + padding,
            center_y + padding,
        ),
        12,
    )

    pygame.draw.line(
        screen,
        color,
        (
            center_x + padding,
            center_y - padding,
        ),
        (
            center_x - padding,
            center_y + padding,
        ),
        12,
    )


def draw_o(
    screen: pygame.Surface,
    center_x: int,
    center_y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw an O mark."""

    pygame.draw.circle(
        screen,
        color,
        (
            center_x,
            center_y,
        ),
        55,
        12,
    )


# ================================================================
# Board
# ================================================================

def draw_board(
    screen: pygame.Surface,
    game: FadingTicTacToe,
    fonts: dict,
    hover_square: int | None = None,
) -> None:
    """Draw the complete game board."""

    board_rect = pygame.Rect(
        BOARD_X,
        BOARD_Y,
        BOARD_SIZE,
        BOARD_SIZE,
    )

    # Board background
    pygame.draw.rect(
        screen,
        BOARD_BACKGROUND,
        board_rect,
        border_radius=10,
    )

    # ------------------------------------------------------------
    # Hover square
    # ------------------------------------------------------------

    if (
        hover_square is not None
        and not game.is_terminal()
        and game.current_player != 0
    ):
        row = hover_square // 3
        col = hover_square % 3

        x = BOARD_X + col * CELL_SIZE
        y = BOARD_Y + row * CELL_SIZE

        # Only highlight legal squares.
        if hover_square in game.legal_moves():

            pygame.draw.rect(
                screen,
                HOVER_COLOR,
                pygame.Rect(
                    x + 4,
                    y + 4,
                    CELL_SIZE - 8,
                    CELL_SIZE - 8,
                ),
                border_radius=8,
            )

    # ------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------

    for i in range(1, 3):

        x = BOARD_X + i * CELL_SIZE

        pygame.draw.line(
            screen,
            GRID_COLOR,
            (
                x,
                BOARD_Y,
            ),
            (
                x,
                BOARD_Y + BOARD_SIZE,
            ),
            5,
        )

        y = BOARD_Y + i * CELL_SIZE

        pygame.draw.line(
            screen,
            GRID_COLOR,
            (
                BOARD_X,
                y,
            ),
            (
                BOARD_X + BOARD_SIZE,
                y,
            ),
            5,
        )

    # ------------------------------------------------------------
    # Board marks
    # ------------------------------------------------------------

    for row in range(3):

        for col in range(3):

            idx = row * 3 + col

            player = game.board[idx]

            center_x = (
                BOARD_X
                + col * CELL_SIZE
                + CELL_SIZE // 2
            )

            center_y = (
                BOARD_Y
                + row * CELL_SIZE
                + CELL_SIZE // 2
            )

            if player == PLAYER_X:

                draw_x(
                    screen,
                    center_x,
                    center_y,
                    X_COLOR,
                )

            elif player == PLAYER_O:

                draw_o(
                    screen,
                    center_x,
                    center_y,
                    O_COLOR,
                )

            else:

                # Empty-cell number
                draw_text(
                    screen,
                    str(idx),
                    fonts["square"],
                    (155, 155, 150),
                    (
                        center_x,
                        center_y,
                    ),
                )


# ================================================================
# Fade indicators
# ================================================================

def draw_fade_indicator(
    screen: pygame.Surface,
    game: FadingTicTacToe,
) -> None:
    """
    Highlight the oldest mark for a player when their next move
    will cause that mark to fade.
    """

    for player in (PLAYER_X, PLAYER_O):

        if len(game.history[player]) >= game.max_active_marks:

            oldest = game.history[player][0]

            row = oldest // 3
            col = oldest % 3

            x = BOARD_X + col * CELL_SIZE
            y = BOARD_Y + row * CELL_SIZE

            pygame.draw.rect(
                screen,
                FADE_COLOR,
                pygame.Rect(
                    x + 12,
                    y + 12,
                    CELL_SIZE - 24,
                    CELL_SIZE - 24,
                ),
                width=5,
                border_radius=10,
            )


# ================================================================
# Last faded square
# ================================================================

def draw_last_faded(
    screen: pygame.Surface,
    last_faded: int | None,
    fonts: dict,
) -> None:
    """Display the last square that faded."""

    if last_faded is None:
        return

    text = f"Square {last_faded} just faded"

    draw_text(
        screen,
        text,
        fonts["fade"],
        FADE_COLOR,
        (
            WINDOW_WIDTH // 2,
            755,
        ),
    )


# ================================================================
# Status
# ================================================================

def draw_status(
    screen: pygame.Surface,
    game: FadingTicTacToe,
    human_player: int,
    agent_thinking: bool,
    fonts: dict,
) -> None:
    """Draw the game status."""

    human_symbol = (
        "X"
        if human_player == PLAYER_X
        else "O"
    )

    agent_symbol = (
        "O"
        if human_player == PLAYER_X
        else "X"
    )

    # ------------------------------------------------------------
    # Main status
    # ------------------------------------------------------------

    if game.is_terminal():

        winner = game.winner()

        if winner == 0:

            message = "Draw"
            color = TEXT_COLOR

        elif winner == human_player:

            message = "You Win!"
            color = WIN_COLOR

        else:

            message = "Agent Wins!"
            color = INVALID_COLOR

    elif agent_thinking:

        message = "Agent is thinking..."
        color = TEXT_COLOR

    elif game.current_player == human_player:

        message = "Your Turn"

        color = (
            X_COLOR
            if human_player == PLAYER_X
            else O_COLOR
        )

    else:

        message = "Agent's Turn"

        color = (
            O_COLOR
            if human_player == PLAYER_X
            else X_COLOR
        )

    draw_text(
        screen,
        message,
        fonts["status"],
        color,
        (
            WINDOW_WIDTH // 2,
            48,
        ),
    )

    # ------------------------------------------------------------
    # Player information
    # ------------------------------------------------------------

    draw_text(
        screen,
        f"You: {human_symbol}    Agent: {agent_symbol}",
        fonts["info"],
        TEXT_COLOR,
        (
            WINDOW_WIDTH // 2,
            88,
        ),
    )


# ================================================================
# Button
# ================================================================

def draw_button(
    screen: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    font: pygame.font.Font,
    mouse_position: tuple[int, int],
) -> None:
    """Draw a button."""

    if rect.collidepoint(mouse_position):
        color = BUTTON_HOVER_COLOR
    else:
        color = BUTTON_COLOR

    pygame.draw.rect(
        screen,
        color,
        rect,
        border_radius=8,
    )

    draw_text(
        screen,
        text,
        font,
        BUTTON_TEXT_COLOR,
        rect.center,
    )


# ================================================================
# Mouse -> board square
# ================================================================

def square_from_mouse(
    pos: tuple[int, int],
) -> int | None:
    """Convert mouse coordinates to a board square."""

    x, y = pos

    if not (
        BOARD_X <= x < BOARD_X + BOARD_SIZE
        and BOARD_Y <= y < BOARD_Y + BOARD_SIZE
    ):
        return None

    col = (
        x - BOARD_X
    ) // CELL_SIZE

    row = (
        y - BOARD_Y
    ) // CELL_SIZE

    return row * 3 + col


# ================================================================
# Reset game
# ================================================================

def new_game(
    config,
) -> FadingTicTacToe:
    """Create a fresh game."""

    return FadingTicTacToe(config)


# ================================================================
# Main
# ================================================================

def main() -> None:

    # ------------------------------------------------------------
    # Arguments
    # ------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Play Fading-Move Tic-Tac-Toe "
            "against an AlphaZero agent."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained checkpoint.",
    )

    parser.add_argument(
        "--human-plays",
        type=str,
        default="X",
        choices=["X", "O"],
        help="Which side the human plays.",
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=400,
        help="MCTS simulations per agent move.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # Random generator
    # ------------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    # ------------------------------------------------------------
    # Load AlphaZero agent
    # ------------------------------------------------------------

    agent = load_agent(
        args.checkpoint,
        device,
        args.simulations,
    )

    # ------------------------------------------------------------
    # Create game
    # ------------------------------------------------------------

    game = new_game(
        agent.config
    )

    human_player = (
        PLAYER_X
        if args.human_plays == "X"
        else PLAYER_O
    )

    # ------------------------------------------------------------
    # Initialize Pygame
    # ------------------------------------------------------------

    pygame.init()

    # Explicitly initialize fonts.
    pygame.font.init()

    screen = pygame.display.set_mode(
        (
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        )
    )

    pygame.display.set_caption(
        "Fading-Move Tic-Tac-Toe"
    )

    clock = pygame.time.Clock()

    fonts = load_fonts()

    # ------------------------------------------------------------
    # Restart button
    # ------------------------------------------------------------

    restart_button = pygame.Rect(
        250,
        780,
        200,
        40,
    )

    # ------------------------------------------------------------
    # State
    # ------------------------------------------------------------

    running = True

    hover_square = None

    last_faded = None

    agent_thinking = False

    # If human chose O, X starts and therefore the agent needs
    # to make the first move.
    agent_move_pending = (
        game.current_player != human_player
    )

    # Used to prevent accidental repeated agent moves.
    agent_move_done = False

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------

    while running:

        # ========================================================
        # Events
        # ========================================================

        for event in pygame.event.get():

            # ----------------------------------------------------
            # Close window
            # ----------------------------------------------------

            if event.type == pygame.QUIT:

                running = False

            # ----------------------------------------------------
            # Mouse movement
            # ----------------------------------------------------

            elif event.type == pygame.MOUSEMOTION:

                hover_square = square_from_mouse(
                    event.pos
                )

            # ----------------------------------------------------
            # Mouse click
            # ----------------------------------------------------

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button != 1:
                    continue

                # ------------------------------------------------
                # Restart
                # ------------------------------------------------

                if restart_button.collidepoint(
                    event.pos
                ):

                    game = new_game(
                        agent.config
                    )

                    last_faded = None

                    agent_thinking = False

                    agent_move_done = False

                    agent_move_pending = (
                        game.current_player
                        != human_player
                    )

                    continue

                # ------------------------------------------------
                # Ignore input while agent thinks
                # ------------------------------------------------

                if agent_thinking:
                    continue

                # ------------------------------------------------
                # Ignore clicks after game ends
                # ------------------------------------------------

                if game.is_terminal():
                    continue

                # ------------------------------------------------
                # Human's turn
                # ------------------------------------------------

                if (
                    game.current_player
                    == human_player
                ):

                    move = square_from_mouse(
                        event.pos
                    )

                    if move is None:
                        continue

                    legal_moves = game.legal_moves()

                    if move not in legal_moves:
                        continue

                    # Make human move.
                    game.make_move(move)

                    last_faded = (
                        game.last_faded_square
                    )

                    # Agent gets next turn.
                    if not game.is_terminal():

                        agent_move_pending = True

                        agent_move_done = False

        # ========================================================
        # Agent move
        # ========================================================

        if (
            agent_move_pending
            and not agent_move_done
            and not game.is_terminal()
            and game.current_player != human_player
        ):

            agent_thinking = True

            # ----------------------------------------------------
            # Show thinking state BEFORE MCTS.
            # ----------------------------------------------------

            screen.fill(
                BACKGROUND
            )

            draw_status(
                screen,
                game,
                human_player,
                True,
                fonts,
            )

            draw_board(
                screen,
                game,
                fonts,
                None,
            )

            draw_fade_indicator(
                screen,
                game,
            )

            draw_last_faded(
                screen,
                last_faded,
                fonts,
            )

            draw_button(
                screen,
                restart_button,
                "Restart",
                fonts["button"],
                pygame.mouse.get_pos(),
            )

            pygame.display.flip()

            # ----------------------------------------------------
            # Ask AlphaZero agent for move.
            # ----------------------------------------------------

            try:

                move = agent.choose_action(
                    game,
                    rng,
                )

            except Exception as exc:

                print(
                    "Error while agent was choosing a move:"
                )
                print(exc)

                agent_thinking = False
                agent_move_pending = False

                continue

            # ----------------------------------------------------
            # Make agent move.
            # ----------------------------------------------------

            game.make_move(move)

            last_faded = (
                game.last_faded_square
            )

            agent_move_done = True
            agent_move_pending = False
            agent_thinking = False

        # ========================================================
        # Drawing
        # ========================================================

        screen.fill(
            BACKGROUND
        )

        draw_status(
            screen,
            game,
            human_player,
            agent_thinking,
            fonts,
        )

        # Only show mouse hover during human's turn.
        board_hover = None

        if (
            not game.is_terminal()
            and not agent_thinking
            and game.current_player == human_player
        ):
            board_hover = hover_square

        draw_board(
            screen,
            game,
            fonts,
            board_hover,
        )

        # Fade warnings
        if not game.is_terminal():
            draw_fade_indicator(
                screen,
                game,
            )

        # Last faded square
        draw_last_faded(
            screen,
            last_faded,
            fonts,
        )

        # Restart button
        draw_button(
            screen,
            restart_button,
            "Restart",
            fonts["button"],
            pygame.mouse.get_pos(),
        )

        pygame.display.flip()

        clock.tick(FPS)

    # ------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------

    pygame.quit()


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    main()
