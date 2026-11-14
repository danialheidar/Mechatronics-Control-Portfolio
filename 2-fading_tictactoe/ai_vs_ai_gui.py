"""
ai_vs_ai_gui.py
===============

Pygame GUI demonstration of AlphaZero AI vs AlphaZero AI
for Fading-Move Tic-Tac-Toe.

Both players use the same trained AlphaZero agent/network.
The game automatically plays one move at a time so that the
match can be visually demonstrated.

Usage:

    python ai_vs_ai_gui.py --checkpoint checkpoints/best.pt

    python ai_vs_ai_gui.py \
        --checkpoint checkpoints/best.pt \
        --simulations 400

    python ai_vs_ai_gui.py \
        --checkpoint checkpoints/best.pt \
        --simulations 200 \
        --delay 800

Controls:

    SPACE       Pause / Resume
    R           Restart game
    ESC         Quit
    UP          Increase delay
    DOWN        Decrease delay

Delay is specified in milliseconds.

Example:

    python ai_vs_ai_gui.py --checkpoint checkpoints/best.pt --delay 1000
"""

from __future__ import annotations

import argparse
import os
import time

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

FADE_COLOR = (220, 170, 50)

BUTTON_COLOR = (65, 65, 65)
BUTTON_HOVER_COLOR = (85, 85, 85)
BUTTON_TEXT_COLOR = (255, 255, 255)

WIN_COLOR = (40, 150, 70)
LOSS_COLOR = (180, 60, 60)


# ================================================================
# Font handling
# ================================================================

def load_font(size: int) -> pygame.font.Font:
    """
    Load a font directly from Windows.

    We avoid pygame.font.SysFont() because some pygame-ce versions
    can produce an UnboundLocalError with SysFont on Windows.
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
                return pygame.font.Font(
                    path,
                    size,
                )
            except pygame.error:
                pass

    # Final fallback.
    return pygame.font.Font(
        None,
        size,
    )


def load_fonts() -> dict[str, pygame.font.Font]:

    return {
        "title": load_font(40),
        "status": load_font(34),
        "info": load_font(24),
        "square": load_font(26),
        "fade": load_font(23),
        "button": load_font(24),
        "result": load_font(46),
        "small": load_font(20),
    }


# ================================================================
# Load AI
# ================================================================

def load_agent(
    checkpoint_path: str | None,
    device: torch.device,
    num_simulations: int,
) -> NetworkAgent:

    if checkpoint_path is None:

        print(
            "WARNING: No checkpoint supplied."
        )

        print(
            "Using a randomly initialized network."
        )

        config = Config()

        network = PolicyValueNet(
            config
        ).to(device)

        network.eval()

        return NetworkAgent(
            network,
            config,
            device,
            num_simulations=num_simulations,
            temperature=10,
        )

    print(
        f"Loading checkpoint: {checkpoint_path}"
    )

    from train import load_checkpoint

    network, _optimizer, _iteration, config = (
        load_checkpoint(
            checkpoint_path,
            device,
        )
    )

    network.eval()

    print(
        "Checkpoint loaded."
    )

    return NetworkAgent(
        network,
        config,
        device,
        num_simulations=num_simulations,
        temperature=0.0,
    )


# ================================================================
# Text
# ================================================================

def draw_text(
    surface: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
    center: tuple[int, int],
) -> None:

    rendered = font.render(
        text,
        True,
        color,
    )

    rect = rendered.get_rect(
        center=center
    )

    surface.blit(
        rendered,
        rect,
    )


# ================================================================
# Draw X
# ================================================================

def draw_x(
    screen: pygame.Surface,
    center_x: int,
    center_y: int,
) -> None:

    padding = 45

    pygame.draw.line(
        screen,
        X_COLOR,
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
        X_COLOR,
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


# ================================================================
# Draw O
# ================================================================

def draw_o(
    screen: pygame.Surface,
    center_x: int,
    center_y: int,
) -> None:

    pygame.draw.circle(
        screen,
        O_COLOR,
        (
            center_x,
            center_y,
        ),
        55,
        12,
    )


# ================================================================
# Draw board
# ================================================================

def draw_board(
    screen: pygame.Surface,
    game: FadingTicTacToe,
    fonts: dict,
) -> None:

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
    # Marks
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
                )

            elif player == PLAYER_O:

                draw_o(
                    screen,
                    center_x,
                    center_y,
                )

            else:

                # Show square number.
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
# Fade indicator
# ================================================================

def draw_fade_indicator(
    screen: pygame.Surface,
    game: FadingTicTacToe,
) -> None:

    for player in (
        PLAYER_X,
        PLAYER_O,
    ):

        if (
            len(game.history[player])
            >= game.max_active_marks
        ):

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

    if last_faded is None:
        return

    draw_text(
        screen,
        f"Square {last_faded} just faded",
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
    thinking: bool,
    paused: bool,
    fonts: dict,
) -> None:

    # ------------------------------------------------------------
    # Main status
    # ------------------------------------------------------------

    if game.is_terminal():

        winner = game.winner()

        if winner == 0:

            message = "DRAW"
            color = TEXT_COLOR

        elif winner == PLAYER_X:

            message = "X WINS!"
            color = X_COLOR

        else:

            message = "O WINS!"
            color = O_COLOR

    elif paused:

        message = "PAUSED"
        color = TEXT_COLOR

    elif thinking:

        player_name = (
            "X"
            if game.current_player == PLAYER_X
            else "O"
        )

        message = (
            f"{player_name} is thinking..."
        )

        color = (
            X_COLOR
            if game.current_player == PLAYER_X
            else O_COLOR
        )

    else:

        player_name = (
            "X"
            if game.current_player == PLAYER_X
            else "O"
        )

        message = (
            f"{player_name}'s turn"
        )

        color = (
            X_COLOR
            if game.current_player == PLAYER_X
            else O_COLOR
        )

    draw_text(
        screen,
        message,
        fonts["status"],
        color,
        (
            WINDOW_WIDTH // 2,
            45,
        ),
    )

    # ------------------------------------------------------------
    # Match information
    # ------------------------------------------------------------

    draw_text(
        screen,
        "AI X     vs     AI O",
        fonts["info"],
        TEXT_COLOR,
        (
            WINDOW_WIDTH // 2,
            85,
        ),
    )


# ================================================================
# Bottom information
# ================================================================

def draw_controls(
    screen: pygame.Surface,
    fonts: dict,
    delay: int,
    paused: bool,
) -> None:

    state = (
        "Paused"
        if paused
        else "Playing"
    )

    draw_text(
        screen,
        f"{state}    |    Delay: {delay} ms",
        fonts["small"],
        TEXT_COLOR,
        (
            WINDOW_WIDTH // 2,
            720,
        ),
    )

    draw_text(
        screen,
        "SPACE: Pause    R: Restart    ↑/↓: Speed    ESC: Quit",
        fonts["small"],
        TEXT_COLOR,
        (
            WINDOW_WIDTH // 2,
            745,
        ),
    )


# ================================================================
# Restart button
# ================================================================

def draw_button(
    screen: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    font: pygame.font.Font,
    mouse_pos: tuple[int, int],
) -> None:

    if rect.collidepoint(mouse_pos):

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
# Main
# ================================================================

def main() -> None:

    # ------------------------------------------------------------
    # Arguments
    # ------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "AI vs AI Pygame demonstration "
            "for Fading-Move Tic-Tac-Toe."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint.",
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=400,
        help="MCTS simulations per move.",
    )

    parser.add_argument(
        "--delay",
        type=int,
        default=700,
        help=(
            "Delay between AI moves in milliseconds."
        ),
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

    print(
        f"Using device: {device}"
    )

    # ------------------------------------------------------------
    # Random generator
    # ------------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    # ------------------------------------------------------------
    # Load AI
    # ------------------------------------------------------------

    agent = load_agent(
        args.checkpoint,
        device,
        args.simulations,
    )

    # ------------------------------------------------------------
    # Initial game
    # ------------------------------------------------------------

    game = FadingTicTacToe(
        agent.config
    )

    # ------------------------------------------------------------
    # Pygame
    # ------------------------------------------------------------

    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode(
        (
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        )
    )

    pygame.display.set_caption(
        "Fading-Move Tic-Tac-Toe - AI vs AI"
    )

    clock = pygame.time.Clock()

    fonts = load_fonts()

    # ------------------------------------------------------------
    # Restart button
    # ------------------------------------------------------------

    restart_button = pygame.Rect(
        250,
        785,
        200,
        38,
    )

    # ------------------------------------------------------------
    # State
    # ------------------------------------------------------------

    running = True

    paused = False

    thinking = False

    last_faded = None

    # Time when the previous move was made.
    last_move_time = time.perf_counter()

    # ------------------------------------------------------------
    # Move delay
    # ------------------------------------------------------------

    delay = max(
        0,
        args.delay,
    )

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------

    while running:

        # ========================================================
        # Events
        # ========================================================

        for event in pygame.event.get():

            # ----------------------------------------------------
            # Window close
            # ----------------------------------------------------

            if event.type == pygame.QUIT:

                running = False

            # ----------------------------------------------------
            # Keyboard
            # ----------------------------------------------------

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:

                    running = False

                elif event.key == pygame.K_SPACE:

                    if not game.is_terminal():

                        paused = not paused

                elif event.key == pygame.K_r:

                    game = FadingTicTacToe(
                        agent.config
                    )

                    last_faded = None

                    thinking = False

                    last_move_time = (
                        time.perf_counter()
                    )

                elif event.key == pygame.K_UP:

                    delay += 100

                    delay = min(
                        delay,
                        5000,
                    )

                    print(
                        f"Delay: {delay} ms"
                    )

                elif event.key == pygame.K_DOWN:

                    delay -= 100

                    delay = max(
                        delay,
                        0,
                    )

                    print(
                        f"Delay: {delay} ms"
                    )

            # ----------------------------------------------------
            # Mouse
            # ----------------------------------------------------

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:

                    if restart_button.collidepoint(
                        event.pos
                    ):

                        game = FadingTicTacToe(
                            agent.config
                        )

                        last_faded = None

                        thinking = False

                        last_move_time = (
                            time.perf_counter()
                        )

        # ========================================================
        # AI move
        # ========================================================

        if (
            not paused
            and not game.is_terminal()
            and not thinking
        ):

            current_time = (
                time.perf_counter()
            )

            elapsed_ms = (
                current_time
                - last_move_time
            ) * 1000

            if elapsed_ms >= delay:

                # ------------------------------------------------
                # Thinking state
                # ------------------------------------------------

                thinking = True

                # Draw the board immediately so the user can see
                # that the AI is thinking.
                screen.fill(
                    BACKGROUND
                )

                draw_status(
                    screen,
                    game,
                    thinking=True,
                    paused=False,
                    fonts=fonts,
                )

                draw_board(
                    screen,
                    game,
                    fonts,
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

                draw_controls(
                    screen,
                    fonts,
                    delay,
                    paused,
                )

                draw_button(
                    screen,
                    restart_button,
                    "Restart",
                    fonts["button"],
                    pygame.mouse.get_pos(),
                )

                pygame.display.flip()

                # ------------------------------------------------
                # Ask AlphaZero for a move
                # ------------------------------------------------

                try:

                    move = agent.choose_action(
                        game,
                        rng,
                    )

                except Exception as exc:

                    print(
                        "Agent error:"
                    )

                    print(exc)

                    thinking = False

                    continue

                # ------------------------------------------------
                # Make move
                # ------------------------------------------------

                game.make_move(
                    move
                )

                last_faded = (
                    game.last_faded_square
                )

                thinking = False

                last_move_time = (
                    time.perf_counter()
                )

                # ------------------------------------------------
                # Console output
                # ------------------------------------------------

                player = (
                    "O"
                    if game.current_player
                    == PLAYER_X
                    else "X"
                )

                # game.current_player is now the NEXT player,
                # so player above is the player who just moved.

                print(
                    f"Move: {move}"
                )

                if last_faded is not None:

                    print(
                        f"  Faded square: "
                        f"{last_faded}"
                    )

        # ========================================================
        # Draw
        # ========================================================

        screen.fill(
            BACKGROUND
        )

        draw_status(
            screen,
            game,
            thinking,
            paused,
            fonts,
        )

        draw_board(
            screen,
            game,
            fonts,
        )

        # --------------------------------------------------------
        # Fade warning
        # --------------------------------------------------------

        if not game.is_terminal():

            draw_fade_indicator(
                screen,
                game,
            )

        # --------------------------------------------------------
        # Last faded
        # --------------------------------------------------------

        draw_last_faded(
            screen,
            last_faded,
            fonts,
        )

        # --------------------------------------------------------
        # Controls
        # --------------------------------------------------------

        draw_controls(
            screen,
            fonts,
            delay,
            paused,
        )

        # --------------------------------------------------------
        # Restart
        # --------------------------------------------------------

        draw_button(
            screen,
            restart_button,
            "Restart",
            fonts["button"],
            pygame.mouse.get_pos(),
        )

        # --------------------------------------------------------
        # Display
        # --------------------------------------------------------

        pygame.display.flip()

        clock.tick(FPS)

    # ============================================================
    # Shutdown
    # ============================================================

    pygame.quit()


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    main()
