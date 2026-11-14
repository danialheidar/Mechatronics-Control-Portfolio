"""
game.py
=======

Self-contained rules engine for **Fading-Move Tic-Tac-Toe**.

Rules recap
------------
* Standard 3x3 board, players X (+1) and O (-1) alternate turns.
* Each player may have at most ``max_active_marks`` (3 by default) marks
  on the board at once. When a player makes a move that would give them
  more than that, their *oldest* mark is removed ("fades") first, and
  the move is then placed.
* A player wins the instant that, after their move **and** the
  resulting fade, they have three active marks in a line (row, column,
  or diagonal).
* Because marks disappear, the same board configuration can recur
  (the game is not obviously acyclic). We detect this by hashing the
  *complete* game state (see ``_state_signature``) and declaring a draw
  if a state repeats ``repetition_draw_count`` times. A hard ply cap
  (``max_game_length``) is kept purely as a defensive fallback and
  should never actually be hit.

Canonical state representation
-------------------------------
The *only* information needed to fully reconstruct the board, the age
of every mark, and whose turn it is, is:

    (tuple(history[+1]), tuple(history[-1]), current_player)

where ``history[p]`` is the ordered list of squares player ``p`` has
placed marks on, oldest first, truncated to the last
``max_active_marks`` entries. The board occupancy is simply the union
of both histories; the *age* of a mark is its position within its
owner's history list. This tuple is used both as the repetition-
detection key and as the basis for the neural-network input encoding
(see ``to_tensor``), because it is provably a sufficient statistic for
the (Markovian) game state -- nothing else needs to be tracked.

A flat 3x3 grid of {X, O, empty} would *not* be sufficient: it tells
you what is on the board right now, but not which mark is about to
fade next for each player, which is essential information for correct
play (e.g. you may want to avoid completing a line that will
immediately un-complete itself, or you may want to force an opponent's
useful mark to fade).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Config

# The 8 winning lines (rows, columns, diagonals) as index triples into
# a flattened 3x3 board (row-major: square = row * 3 + col).
LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
)

PLAYER_X = 1
PLAYER_O = -1
EMPTY = 0

_SYMBOL = {PLAYER_X: "X", PLAYER_O: "O", EMPTY: "."}


@dataclass(frozen=True)
class StateSnapshot:
    """An immutable, easily-serializable view of a game state."""

    board: Tuple[int, ...]                    # length 9, values in {-1,0,1}
    history_x: Tuple[int, ...]                # oldest -> newest
    history_o: Tuple[int, ...]                # oldest -> newest
    current_player: int
    done: bool
    winner: Optional[int]                     # 1, -1, 0 (draw) or None
    ply: int


class FadingTicTacToe:
    """Rules engine for Fading-Move Tic-Tac-Toe.

    This class has *no* knowledge of neural networks or search; it only
    implements the game rules, so it can be unit tested in isolation and
    reused unmodified by self-play, evaluation, and the human-play CLI.
    """

    def __init__(self, config: Optional[Config] = None):
        cfg = config or Config()
        self.max_active_marks: int = cfg.max_active_marks
        self.repetition_draw_count: int = cfg.repetition_draw_count
        self.max_game_length: int = cfg.max_game_length
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.board: List[int] = [EMPTY] * 9
        self.history: Dict[int, List[int]] = {PLAYER_X: [], PLAYER_O: []}
        self.current_player: int = PLAYER_X
        self.done: bool = False
        self.winner_: Optional[int] = None
        self.ply: int = 0
        self.last_faded_square: Optional[int] = None
        self.last_moved_player: Optional[int] = None
        self._state_counts: Dict[Tuple, int] = {}
        self._register_state()

    def clone(self) -> "FadingTicTacToe":
        """Deep-enough copy for use as a fresh MCTS/self-play branch."""
        new = FadingTicTacToe.__new__(FadingTicTacToe)
        new.max_active_marks = self.max_active_marks
        new.repetition_draw_count = self.repetition_draw_count
        new.max_game_length = self.max_game_length
        new.board = list(self.board)
        new.history = {PLAYER_X: list(self.history[PLAYER_X]),
                        PLAYER_O: list(self.history[PLAYER_O])}
        new.current_player = self.current_player
        new.done = self.done
        new.winner_ = self.winner_
        new.ply = self.ply
        new.last_faded_square = self.last_faded_square
        new.last_moved_player = self.last_moved_player
        new._state_counts = dict(self._state_counts)
        return new

    # ------------------------------------------------------------------
    # Core rules
    # ------------------------------------------------------------------
    def legal_moves(self) -> List[int]:
        if self.done:
            return []
        return [i for i in range(9) if self.board[i] == EMPTY]

    def make_move(self, action: int) -> None:
        """Apply ``action`` (a square index 0..8) for the current player."""
        if self.done:
            raise ValueError("Cannot move: game has already terminated.")
        if not (0 <= action < 9):
            raise ValueError(f"Illegal action {action}: out of range 0..8.")
        if self.board[action] != EMPTY:
            raise ValueError(f"Illegal move: square {action} is occupied.")

        player = self.current_player
        self.board[action] = player
        self.history[player].append(action)
        self.last_moved_player = player

        faded_square: Optional[int] = None
        if len(self.history[player]) > self.max_active_marks:
            faded_square = self.history[player].pop(0)
            # The oldest recorded move for this player must still be
            # exactly their mark on the board -- a broken invariant here
            # would indicate a bug in fading/turn bookkeeping.
            assert self.board[faded_square] == player, (
                "Internal invariant violated: fading square did not "
                "belong to the player whose move history it came from."
            )
            self.board[faded_square] = EMPTY
        self.last_faded_square = faded_square
        self.ply += 1

        # Turn always passes to the opponent, even on a terminal move.
        # (Keeping this unconditional -- rather than only switching on
        # non-terminal moves -- means "current_player" always denotes
        # "the side that would move next", which is the convention MCTS
        # relies on for consistent value-perspective alternation when
        # backing up through terminal leaves.)
        self.current_player = -player

        # Winning is checked *after* fading has been applied, using only
        # this player's currently active marks.
        if self._has_line(player):
            self.done = True
            self.winner_ = player
            return

        # Repetition-based draw detection. Note that with
        # max_active_marks == 3 per side, at most 6 of the 9 squares can
        # ever be occupied simultaneously, so at least 3 squares are
        # always free and the side to move always has a legal move.
        # "No legal moves" can therefore never be a terminal condition
        # in this variant -- draws only ever arise from state repetition
        # (or the defensive ply cap below).
        occurrences = self._register_state()
        if occurrences >= self.repetition_draw_count:
            self.done = True
            self.winner_ = 0
            return

        if self.ply >= self.max_game_length:
            self.done = True
            self.winner_ = 0
            return

    def _has_line(self, player: int) -> bool:
        active = set(self.history[player])
        for a, b, c in LINES:
            if a in active and b in active and c in active:
                return True
        return False

    def is_terminal(self) -> bool:
        return self.done

    def winner(self) -> Optional[int]:
        """Returns 1 (X won), -1 (O won), 0 (draw), or None if not over."""
        return self.winner_

    # ------------------------------------------------------------------
    # State signature / repetition detection
    # ------------------------------------------------------------------
    def _state_signature(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        return (
            tuple(self.history[PLAYER_X]),
            tuple(self.history[PLAYER_O]),
            self.current_player,
        )

    def _register_state(self) -> int:
        sig = self._state_signature()
        count = self._state_counts.get(sig, 0) + 1
        self._state_counts[sig] = count
        return count

    def get_state(self) -> StateSnapshot:
        return StateSnapshot(
            board=tuple(self.board),
            history_x=tuple(self.history[PLAYER_X]),
            history_o=tuple(self.history[PLAYER_O]),
            current_player=self.current_player,
            done=self.done,
            winner=self.winner_,
            ply=self.ply,
        )

    # ------------------------------------------------------------------
    # Neural-network input encoding
    # ------------------------------------------------------------------
    def to_tensor(self) -> np.ndarray:
        """Encode the current state as a ``(7, 3, 3)`` float32 tensor.

        Planes 0-2: player X's marks, ordered by *recency rank* --
            plane 0 = X's most recently placed active mark (age rank 0)
            plane 1 = X's second-most-recent active mark (age rank 1)
            plane 2 = X's oldest active mark (age rank 2), i.e. the mark
                      that will fade on X's *next* move, if X has 3.
            A plane is all-zero if X does not currently have a mark at
            that recency rank (e.g. early in the game).
        Planes 3-5: same, for player O.
        Plane 6: constant plane, all 1.0 if it is X's turn, all 0.0 if
            it is O's turn.

        This encoding is a bijection with the canonical state signature
        used internally (histories + current player), so it loses no
        information relevant to predicting future fading, and it is a
        fixed-size representation regardless of how many marks are on
        the board.
        """
        planes = np.zeros((7, 3, 3), dtype=np.float32)
        for player, base in ((PLAYER_X, 0), (PLAYER_O, 3)):
            # Reverse so index 0 = newest, ..., last = oldest.
            recency_ordered = list(reversed(self.history[player]))
            for age_rank, square in enumerate(recency_ordered):
                r, c = divmod(square, 3)
                planes[base + age_rank, r, c] = 1.0
        if self.current_player == PLAYER_X:
            planes[6, :, :] = 1.0
        return planes

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> str:
        rows = []
        for r in range(3):
            cells = []
            for c in range(3):
                idx = r * 3 + c
                cells.append(_SYMBOL[self.board[idx]])
            rows.append(" ".join(cells))
        board_str = "\n".join(rows)
        about_to_fade = []
        for player, name in ((PLAYER_X, "X"), (PLAYER_O, "O")):
            if len(self.history[player]) >= self.max_active_marks:
                about_to_fade.append(f"{name}'s next move fades square "
                                      f"{self.history[player][0]}")
        info = "\n".join(about_to_fade)
        return board_str + ("\n" + info if info else "")

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.render()
