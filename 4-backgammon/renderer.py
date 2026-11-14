import pygame


# =========================================================
# Colors
# =========================================================

BG_COLOR = (30, 30, 30)

BOARD_COLOR = (181, 132, 85)
BOARD_DARK = (125, 78, 45)
BOARD_LIGHT = (224, 180, 120)

WHITE_CHECKER = (235, 235, 235)
BLACK_CHECKER = (35, 35, 35)

TEXT_COLOR = (245, 245, 245)
BAR_COLOR = (95, 60, 35)

HIGHLIGHT = (255, 210, 60)

BORDER_COLOR = (40, 25, 15)


# =========================================================
# Helpers
# =========================================================

def get_player_count(container, player):
    """
    Read player count from either:

        dict:
            {1: x, -1: y}

    or

        list/tuple:
            index 1 -> WHITE
            index -1 -> BLACK

    This makes the renderer independent from the
    exact container type used by the game engine.
    """

    if isinstance(container, dict):

        return int(
            container.get(
                player,
                0
            )
        )

    return int(
        container[player]
    )


# =========================================================
# Renderer
# =========================================================

class BackgammonRenderer:

    def __init__(
        self,
        width=1200,
        height=800,
        fps=60,
        enabled=True
    ):

        self.width = width
        self.height = height

        self.fps = fps

        self.enabled = enabled

        self.screen = None
        self.clock = None

        self.font = None
        self.small_font = None
        self.big_font = None

        self.running = True

        # -------------------------------------------------
        # Geometry
        # -------------------------------------------------

        self.board_left = 80
        self.board_top = 90

        self.board_width = 1040
        self.board_height = 600

        self.bar_width = 70

        self.point_width = (
            self.board_width / 12
        )

        # -------------------------------------------------
        # Pygame
        # -------------------------------------------------

        if self.enabled:

            pygame.init()

            self.screen = pygame.display.set_mode(
                (
                    self.width,
                    self.height
                )
            )

            pygame.display.set_caption(
                "DQN Backgammon"
            )

            self.clock = pygame.time.Clock()

            self.font = pygame.font.SysFont(
                "Arial",
                24
            )

            self.small_font = pygame.font.SysFont(
                "Arial",
                18
            )

            self.big_font = pygame.font.SysFont(
                "Arial",
                34
            )

    # =====================================================
    # Close
    # =====================================================

    def close(self):

        if not self.enabled:
            return

        self.running = False

        pygame.quit()

    # =====================================================
    # Event handling
    # =====================================================

    def handle_events(self):

        if not self.enabled:
            return

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                self.close()

    # =====================================================
    # Point center
    # =====================================================

    def point_center(
        self,
        point
    ):
        """
        Convert internal point number [0..23]
        to screen coordinates.

        Bottom:
            0 ... 11

        Top:
            23 ... 12
        """

        if point < 0 or point > 23:

            raise ValueError(
                f"Invalid point: {point}"
            )

        # -------------------------------------------------
        # Bottom half
        # -------------------------------------------------

        if point < 12:

            index = point

            x = (
                self.board_left
                +
                index * self.point_width
                +
                self.point_width / 2
            )

            y = (
                self.board_top
                +
                self.board_height
                -
                35
            )

        # -------------------------------------------------
        # Top half
        # -------------------------------------------------

        else:

            index = 23 - point

            x = (
                self.board_left
                +
                index * self.point_width
                +
                self.point_width / 2
            )

            y = (
                self.board_top
                +
                35
            )

        return int(x), int(y)

    # =====================================================
    # Draw checker
    # =====================================================

    def draw_checker(
        self,
        x,
        y,
        is_white,
        radius=27
    ):

        color = (
            WHITE_CHECKER
            if is_white
            else BLACK_CHECKER
        )

        # Shadow
        pygame.draw.circle(
            self.screen,
            (15, 15, 15),
            (
                x + 2,
                y + 3
            ),
            radius
        )

        # Checker
        pygame.draw.circle(
            self.screen,
            color,
            (
                x,
                y
            ),
            radius
        )

        # Outer border
        pygame.draw.circle(
            self.screen,
            BORDER_COLOR,
            (
                x,
                y
            ),
            radius,
            2
        )

        # Inner ring
        pygame.draw.circle(
            self.screen,
            (
                150,
                150,
                150
            ),
            (
                x,
                y
            ),
            radius - 5,
            1
        )

    # =====================================================
    # Draw board
    # =====================================================

    def draw_board(
        self,
        game
    ):

        self.screen.fill(
            BG_COLOR
        )

        # -------------------------------------------------
        # Board rectangle
        # -------------------------------------------------

        pygame.draw.rect(
            self.screen,
            BOARD_COLOR,
            (
                self.board_left,
                self.board_top,
                self.board_width,
                self.board_height
            ),
            border_radius=10
        )

        # -------------------------------------------------
        # Board border
        # -------------------------------------------------

        pygame.draw.rect(
            self.screen,
            BORDER_COLOR,
            (
                self.board_left,
                self.board_top,
                self.board_width,
                self.board_height
            ),
            width=5,
            border_radius=10
        )

        # -------------------------------------------------
        # Middle bar
        # -------------------------------------------------

        center_x = (
            self.board_left
            +
            self.board_width / 2
        )

        pygame.draw.rect(
            self.screen,
            BAR_COLOR,
            (
                int(
                    center_x
                    -
                    self.bar_width / 2
                ),
                self.board_top,
                self.bar_width,
                self.board_height
            )
        )

        # -------------------------------------------------
        # Triangles
        # -------------------------------------------------

        self.draw_triangles()

        # -------------------------------------------------
        # Checkers
        # -------------------------------------------------

        self.draw_checkers(
            game
        )

        # -------------------------------------------------
        # Bar
        # -------------------------------------------------

        self.draw_bar(
            game
        )

        # -------------------------------------------------
        # Text
        # -------------------------------------------------

        self.draw_text(
            game
        )

    # =====================================================
    # Draw triangles
    # =====================================================

    def draw_triangles(self):

        for i in range(12):

            x1 = (
                self.board_left
                +
                i * self.point_width
            )

            x2 = (
                x1
                +
                self.point_width
            )

            center_x = (
                x1 + x2
            ) / 2

            # -------------------------------------------------
            # Top
            # -------------------------------------------------

            top_color = (
                BOARD_DARK
                if i % 2 == 0
                else BOARD_LIGHT
            )

            pygame.draw.polygon(
                self.screen,
                top_color,
                [
                    (
                        int(x1 + 4),
                        self.board_top + 4
                    ),
                    (
                        int(x2 - 4),
                        self.board_top + 4
                    ),
                    (
                        int(center_x),
                        self.board_top + 250
                    )
                ]
            )

            # -------------------------------------------------
            # Bottom
            # -------------------------------------------------

            bottom_color = (
                BOARD_LIGHT
                if i % 2 == 0
                else BOARD_DARK
            )

            pygame.draw.polygon(
                self.screen,
                bottom_color,
                [
                    (
                        int(x1 + 4),
                        self.board_top
                        +
                        self.board_height
                        -
                        4
                    ),
                    (
                        int(x2 - 4),
                        self.board_top
                        +
                        self.board_height
                        -
                        4
                    ),
                    (
                        int(center_x),
                        self.board_top
                        +
                        self.board_height
                        -
                        250
                    )
                ]
            )

    # =====================================================
    # Draw all checkers
    # =====================================================

    def draw_checkers(
        self,
        game
    ):

        for point in range(24):

            value = game.points[point]

            if value == 0:
                continue

            count = abs(
                int(value)
            )

            is_white = (
                value > 0
            )

            x, _ = self.point_center(
                point
            )

            # -------------------------------------------------
            # Bottom points
            # -------------------------------------------------

            if point < 12:

                base_y = (
                    self.board_top
                    +
                    self.board_height
                    -
                    35
                )

                direction = -1

            # -------------------------------------------------
            # Top points
            # -------------------------------------------------

            else:

                base_y = (
                    self.board_top
                    +
                    35
                )

                direction = 1

            # -------------------------------------------------
            # Draw visible checkers
            # -------------------------------------------------

            visible = min(
                count,
                5
            )

            for i in range(
                visible
            ):

                y = (
                    base_y
                    +
                    direction
                    *
                    i
                    *
                    43
                )

                self.draw_checker(
                    x,
                    int(y),
                    is_white
                )

            # -------------------------------------------------
            # Count if stack > 5
            # -------------------------------------------------

            if count > 5:

                label = self.small_font.render(
                    str(count),
                    True,
                    TEXT_COLOR
                )

                label_rect = (
                    label.get_rect(
                        center=(
                            x,
                            int(
                                base_y
                                +
                                direction
                                *
                                225
                            )
                        )
                    )
                )

                self.screen.blit(
                    label,
                    label_rect
                )

    # =====================================================
    # Draw bar
    # =====================================================

    def draw_bar(
        self,
        game
    ):

        center_x = int(
            self.board_left
            +
            self.board_width / 2
        )

        white_count = get_player_count(
            game.bar,
            1
        )

        black_count = get_player_count(
            game.bar,
            -1
        )

        # -------------------------------------------------
        # White bar
        # -------------------------------------------------

        for i in range(
            min(
                white_count,
                5
            )
        ):

            self.draw_checker(
                center_x,
                300 + i * 43,
                True
            )

        # -------------------------------------------------
        # Black bar
        # -------------------------------------------------

        for i in range(
            min(
                black_count,
                5
            )
        ):

            self.draw_checker(
                center_x,
                500 - i * 43,
                False
            )

        # -------------------------------------------------
        # Large count
        # -------------------------------------------------

        if white_count > 5:

            text = self.small_font.render(
                str(white_count),
                True,
                TEXT_COLOR
            )

            rect = text.get_rect(
                center=(
                    center_x,
                    300 + 5 * 43
                )
            )

            self.screen.blit(
                text,
                rect
            )

        if black_count > 5:

            text = self.small_font.render(
                str(black_count),
                True,
                TEXT_COLOR
            )

            rect = text.get_rect(
                center=(
                    center_x,
                    500 - 5 * 43
                )
            )

            self.screen.blit(
                text,
                rect
            )

    # =====================================================
    # Draw point numbers
    # =====================================================

    def draw_point_numbers(self):

        for point in range(24):

            x, _ = self.point_center(
                point
            )

            # Bottom labels
            if point < 12:

                y = (
                    self.board_top
                    +
                    self.board_height
                    +
                    8
                )

            # Top labels
            else:

                y = (
                    self.board_top
                    -
                    30
                )

            label = self.small_font.render(
                str(point + 1),
                True,
                TEXT_COLOR
            )

            rect = label.get_rect(
                center=(
                    x,
                    y
                )
            )

            self.screen.blit(
                label,
                rect
            )

    # =====================================================
    # Draw text information
    # =====================================================

    def draw_text(
        self,
        game
    ):

        # -------------------------------------------------
        # Title
        # -------------------------------------------------

        title = self.big_font.render(
            "DQN Backgammon",
            True,
            TEXT_COLOR
        )

        self.screen.blit(
            title,
            (
                40,
                25
            )
        )

        # -------------------------------------------------
        # Current player
        # -------------------------------------------------

        if game.current_player == 1:

            player_name = "WHITE"

        else:

            player_name = "BLACK"

        player_text = self.font.render(
            f"Current player: {player_name}",
            True,
            TEXT_COLOR
        )

        self.screen.blit(
            player_text,
            (
                720,
                25
            )
        )

        # -------------------------------------------------
        # Dice
        # -------------------------------------------------

        dice_text = self.font.render(
            f"Dice: {game.dice}",
            True,
            TEXT_COLOR
        )

        self.screen.blit(
            dice_text,
            (
                40,
                720
            )
        )

        # -------------------------------------------------
        # Bar
        # -------------------------------------------------

        white_bar = get_player_count(
            game.bar,
            1
        )

        black_bar = get_player_count(
            game.bar,
            -1
        )

        bar_text = self.small_font.render(
            (
                f"Bar - White: {white_bar}   "
                f"Black: {black_bar}"
            ),
            True,
            TEXT_COLOR
        )

        self.screen.blit(
            bar_text,
            (
                300,
                725
            )
        )

        # -------------------------------------------------
        # Borne off
        # -------------------------------------------------

        white_off = get_player_count(
            game.borne_off,
            1
        )

        black_off = get_player_count(
            game.borne_off,
            -1
        )

        off_text = self.small_font.render(
            (
                f"Off - White: {white_off}   "
                f"Black: {black_off}"
            ),
            True,
            TEXT_COLOR
        )

        self.screen.blit(
            off_text,
            (
                700,
                725
            )
        )

        # -------------------------------------------------
        # Point numbers
        # -------------------------------------------------

        self.draw_point_numbers()

        # -------------------------------------------------
        # Winner
        # -------------------------------------------------

        if game.done:

            if game.winner == 1:

                winner = "WHITE"

            else:

                winner = "BLACK"

            text = self.big_font.render(
                f"{winner} WINS!",
                True,
                HIGHLIGHT
            )

            rect = text.get_rect(
                center=(
                    self.width // 2,
                    self.height // 2
                )
            )

            # Background
            background_rect = rect.inflate(
                50,
                30
            )

            pygame.draw.rect(
                self.screen,
                (20, 20, 20),
                background_rect,
                border_radius=10
            )

            self.screen.blit(
                text,
                rect
            )

    # =====================================================
    # Render
    # =====================================================

    def render(
        self,
        game,
        delay=0.0
    ):

        if not self.enabled:
            return

        if not self.running:
            return

        self.handle_events()

        if not self.running:
            return

        self.draw_board(
            game
        )

        pygame.display.flip()

        if delay > 0:

            pygame.time.delay(
                int(
                    delay * 1000
                )
            )

        self.clock.tick(
            self.fps
        )