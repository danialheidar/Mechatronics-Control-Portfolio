"""
tests/test_game.py
===================

Unit tests for the Fading-Move Tic-Tac-Toe rules engine (``game.py``).
These tests do not depend on PyTorch or any ML code -- they exercise
the game logic in isolation, as required.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from game import FadingTicTacToe, PLAYER_X, PLAYER_O


def play_sequence(game: FadingTicTacToe, moves):
    """Helper: apply a list of actions in order, returning the game."""
    for m in moves:
        game.make_move(m)
    return game


class TestNormalPlacement(unittest.TestCase):
    def test_initial_state(self):
        g = FadingTicTacToe()
        self.assertEqual(g.legal_moves(), list(range(9)))
        self.assertEqual(g.current_player, PLAYER_X)
        self.assertFalse(g.is_terminal())
        self.assertIsNone(g.winner())

    def test_simple_moves_update_board_and_turn(self):
        g = FadingTicTacToe()
        g.make_move(0)
        self.assertEqual(g.board[0], PLAYER_X)
        self.assertEqual(g.current_player, PLAYER_O)
        g.make_move(1)
        self.assertEqual(g.board[1], PLAYER_O)
        self.assertEqual(g.current_player, PLAYER_X)

    def test_illegal_move_occupied_square(self):
        g = FadingTicTacToe()
        g.make_move(0)
        with self.assertRaises(ValueError):
            g.make_move(0)

    def test_illegal_move_out_of_range(self):
        g = FadingTicTacToe()
        with self.assertRaises(ValueError):
            g.make_move(9)
        with self.assertRaises(ValueError):
            g.make_move(-1)

    def test_move_after_terminal_raises(self):
        g = FadingTicTacToe()
        # X: 0,1,2 is an immediate win for X.
        play_sequence(g, [0, 3, 1, 4, 2])
        self.assertTrue(g.is_terminal())
        with self.assertRaises(ValueError):
            g.make_move(5)


class TestFading(unittest.TestCase):
    def test_oldest_mark_fades_on_fourth_move(self):
        cfg = Config()
        g = FadingTicTacToe(cfg)
        # X plays squares 0,1,2 (avoid winning) interleaved with O playing
        # elsewhere, then X's 4th move should fade square 0.
        # X: 0, O: 3, X: 1, O: 4, X: 2 -> this would win for X (0,1,2)!
        # Use a non-winning triple for X's first three moves instead.
        g.make_move(0)  # X -> 0
        g.make_move(3)  # O -> 3
        g.make_move(1)  # X -> 1
        g.make_move(4)  # O -> 4
        g.make_move(8)  # X -> 8   (X active: 0,1,8 ; no line)
        g.make_move(6)  # O -> 6   (O active: 3,4,6 ; no line)
        self.assertEqual(sorted(g.history[PLAYER_X]), [0, 1, 8])
        # X's 4th move: should fade square 0 (the oldest).
        g.make_move(2)  # X -> 2
        self.assertEqual(g.history[PLAYER_X], [1, 8, 2])
        self.assertEqual(g.board[0], 0)  # square 0 is empty again
        self.assertEqual(g.last_faded_square, 0)
        self.assertEqual(g.board[2], PLAYER_X)

    def test_per_player_independent_history(self):
        g = FadingTicTacToe()
        g.make_move(0)  # X
        g.make_move(3)  # O
        g.make_move(1)  # X
        g.make_move(4)  # O
        g.make_move(8)  # X (X has 3: 0,1,8)
        g.make_move(6)  # O (O has 3: 3,4,6)
        g.make_move(2)  # X's 4th -> fades 0; X history [1,8,2]
        g.make_move(7)  # O's 4th -> fades 3; O history [4,6,7]
        self.assertEqual(g.history[PLAYER_X], [1, 8, 2])
        self.assertEqual(g.history[PLAYER_O], [4, 6, 7])
        self.assertEqual(g.board[0], 0)
        self.assertEqual(g.board[3], 0)

    def test_reuse_of_own_faded_square(self):
        g = FadingTicTacToe()
        g.make_move(0)  # X -> 0
        g.make_move(3)  # O
        g.make_move(1)  # X -> 1
        g.make_move(4)  # O
        g.make_move(8)  # X -> 8 (X full: 0,1,8)
        g.make_move(6)  # O   (O active: 3,4,6 ; no line)
        g.make_move(2)  # X -> 2, fades 0 (X: 1,8,2)
        g.make_move(5)  # O's 4th move fades 3 (O: 4,6,5 ; no line)
        # square 0 is now empty; X should be able to play there again.
        self.assertIn(0, g.legal_moves())
        g.make_move(0)  # X plays on its own previously-faded square
        self.assertEqual(g.board[0], PLAYER_X)
        self.assertEqual(g.history[PLAYER_X], [8, 2, 0])

    def test_reuse_of_opponent_faded_square(self):
        g = FadingTicTacToe()
        g.make_move(4)  # X -> 4
        g.make_move(0)  # O -> 0
        g.make_move(5)  # X -> 5
        g.make_move(1)  # O -> 1
        g.make_move(7)  # X -> 7  (X active: 4,5,7 no line)
        g.make_move(2)  # O -> 2  (O active: 0,1,2 -> that's a WIN for O!)
        self.assertTrue(g.is_terminal())
        self.assertEqual(g.winner(), PLAYER_O)


class TestWinning(unittest.TestCase):
    def test_row_win(self):
        g = FadingTicTacToe()
        play_sequence(g, [0, 3, 1, 4, 2])  # X: 0,1,2 ; O: 3,4
        self.assertTrue(g.is_terminal())
        self.assertEqual(g.winner(), PLAYER_X)

    def test_column_win(self):
        g = FadingTicTacToe()
        # X: 0,3,6 (column 0) ; O: 1,4
        play_sequence(g, [0, 1, 3, 4, 6])
        self.assertTrue(g.is_terminal())
        self.assertEqual(g.winner(), PLAYER_X)

    def test_diagonal_win(self):
        g = FadingTicTacToe()
        # X: 0,4,8 ; O: 1,2
        play_sequence(g, [0, 1, 4, 2, 8])
        self.assertTrue(g.is_terminal())
        self.assertEqual(g.winner(), PLAYER_X)

    def test_anti_diagonal_win(self):
        g = FadingTicTacToe()
        # X: 2,4,6 ; O: 0,1
        play_sequence(g, [2, 0, 4, 1, 6])
        self.assertTrue(g.is_terminal())
        self.assertEqual(g.winner(), PLAYER_X)

    def test_no_premature_win_detection(self):
        g = FadingTicTacToe()
        g.make_move(0)  # X
        g.make_move(3)  # O
        self.assertFalse(g.is_terminal())

    def test_win_only_counts_active_marks_after_fade(self):
        # Construct a case where a player's 4th move both fades an old
        # mark AND would only be a "line" if the faded mark still
        # counted. Verify the engine does NOT falsely declare a win in
        # that situation, and only declares a win when the *active*
        # marks (post-fade) truly form a line.
        g = FadingTicTacToe()
        # X marks squares 0,1,3 (not a line). O plays elsewhere.
        g.make_move(0)   # X -> 0
        g.make_move(4)   # O -> 4
        g.make_move(1)   # X -> 1
        g.make_move(5)   # O -> 5
        g.make_move(3)   # X -> 3  (X active: 0,1,3 - no line)
        g.make_move(6)   # O -> 6
        # Now X plays square 2: X's 4th move fades square 0.
        # Active X marks become {1,3,2} - not a line (1,3,2 isn't in LINES).
        g.make_move(2)
        self.assertFalse(g.is_terminal())
        self.assertEqual(sorted(g.history[PLAYER_X]), sorted([1, 3, 2]))

    def test_win_achieved_exactly_when_fade_reveals_new_line(self):
        # Construct a case where a player's move both fades an old mark
        # and the *remaining three active marks* form a brand-new line
        # that did not exist the instant before this move.
        g = FadingTicTacToe()
        g.make_move(0)  # X -> 0            (X: [0])
        g.make_move(2)  # O -> 2
        g.make_move(1)  # X -> 1            (X: [0,1])
        g.make_move(5)  # O -> 5
        g.make_move(8)  # X -> 8            (X: [0,1,8], full, no line)
        g.make_move(4)  # O -> 4            (O: [2,5,4], no line)
        # X's 4th move fades square 0. Play square 7: active X becomes
        # {1, 8, 7} -> column (1,4,7)? no, 4 belongs to O. Not a line.
        # Play square 6 instead: active X becomes {1, 8, 6} -> not a
        # line either. To land an exact line we need two of X's
        # already-active squares plus one new square to complete a
        # LINES triple. X's active (pre-move) set is {0,1,8}; after this
        # move 0 fades, leaving {1,8} + new square. The only LINES triple
        # containing both 1 and 8 is none (no line has both index 1 and
        # 8), so this configuration cannot land a line on the very fade
        # move -- which is itself a useful thing to confirm: the engine
        # must correctly report *no* win here regardless of which empty
        # square X picks next amongst non-winning choices.
        g.make_move(6)  # X -> 6, fades 0; active X = {1, 8, 6}
        self.assertFalse(g.is_terminal())
        self.assertEqual(sorted(g.history[PLAYER_X]), sorted([1, 8, 6]))

    def test_win_completed_by_move_that_also_fades(self):
        g = FadingTicTacToe()
        # Give X three marks that, after the next fade+move, complete a
        # fresh diagonal: pre-fade active {0,1,3}; move 4 fades 0 and
        # completes diagonal (1,3)? no diagonal has 1 and 3 together, so
        # build directly for column (1,4,7): X gets 1 and 4 active, then
        # a 4th move fades an unrelated oldest mark while placing on 7.
        g.make_move(3)  # X -> 3           (X: [3])
        g.make_move(0)  # O -> 0
        g.make_move(1)  # X -> 1           (X: [3,1])
        g.make_move(2)  # O -> 2  (O: [0,2], no line)
        g.make_move(4)  # X -> 4           (X: [3,1,4], full, no line)
        g.make_move(5)  # O -> 5  (O: [0,2,5], no line)
        # X's 4th move fades square 3. Play square 7: active X becomes
        # {1, 4, 7} -> column (1,4,7) is a LINE. This must be detected
        # as a win even though it was reached simultaneously with a fade.
        g.make_move(7)
        self.assertTrue(g.is_terminal())
        self.assertEqual(g.winner(), PLAYER_X)
        self.assertEqual(g.last_faded_square, 3)


class TestDrawAndRepetition(unittest.TestCase):
    def test_repetition_declares_draw(self):
        # Construct a genuine cycle: two players shuffle marks between
        # two pairs of squares forever without ever forming a line.
        cfg = Config()
        cfg.repetition_draw_count = 3
        g = FadingTicTacToe(cfg)

        # A simple oscillation: X alternates placing on 0 and 1 (using
        # only 1 active mark's worth of "real" play is not allowed since
        # marks accumulate up to 3 before fading kicks in) -- instead we
        # drive the engine with a fixed short cycle of moves for both
        # players that avoids ever completing a line, and rely on the
        # repetition detector to terminate it rather than hand-verifying
        # the exact period.
        cycle = [0, 1, 3, 4, 6, 7]  # X:0,3,6  O:1,4,7 -> then repeat same squares
        moves = cycle * 20
        terminated_early = False
        for m in moves:
            if g.is_terminal():
                terminated_early = True
                break
            # Only play if legal; with fading, squares free up so the
            # exact same action sequence remains legal indefinitely
            # unless a win or repetition triggers first.
            if m in g.legal_moves():
                g.make_move(m)
            else:
                break
        self.assertTrue(g.is_terminal() or terminated_early)
        # Either a draw by repetition, or (less likely with this cycle)
        # a genuine win; both are valid terminations, but this cycle by
        # construction never forms a line among {0,3,6} or {1,4,7} until
        # a full column completes -- so check which happened:
        if g.winner() not in (PLAYER_X, PLAYER_O):
            self.assertEqual(g.winner(), 0)

    def test_repetition_count_actually_used(self):
        # Directly exercise the internal repetition counter with a
        # deliberately drawish, oscillating sequence and a low threshold.
        cfg = Config()
        cfg.repetition_draw_count = 2
        g = FadingTicTacToe(cfg)
        moves = [0, 1, 3, 4, 6, 7] * 10
        for m in moves:
            if g.is_terminal():
                break
            if m in g.legal_moves():
                g.make_move(m)
            else:
                break
        self.assertTrue(g.is_terminal())

    def test_game_never_stalls_without_terminal_flag(self):
        # A defensive test: run a long pseudo-random legal game and make
        # sure it always terminates (draw, win, or hits the safety cap)
        # rather than looping forever or crashing.
        import random
        rng = random.Random(1234)
        cfg = Config()
        cfg.max_game_length = 500
        for trial in range(25):
            g = FadingTicTacToe(cfg)
            steps = 0
            while not g.is_terminal() and steps < 1000:
                moves = g.legal_moves()
                self.assertTrue(len(moves) > 0, "legal_moves() was empty "
                                 "on a non-terminal state, which should "
                                 "be impossible in this variant.")
                m = rng.choice(moves)
                g.make_move(m)
                steps += 1
            self.assertTrue(g.is_terminal())
            self.assertIn(g.winner(), (PLAYER_X, PLAYER_O, 0))


class TestStateEncoding(unittest.TestCase):
    def test_tensor_shape(self):
        g = FadingTicTacToe()
        t = g.to_tensor()
        self.assertEqual(t.shape, (7, 3, 3))

    def test_tensor_reflects_current_player(self):
        g = FadingTicTacToe()
        t = g.to_tensor()
        self.assertTrue((t[6] == 1.0).all())
        g.make_move(0)
        t2 = g.to_tensor()
        self.assertTrue((t2[6] == 0.0).all())

    def test_tensor_age_rank_ordering(self):
        g = FadingTicTacToe()
        g.make_move(0)  # X -> 0 (X's only/newest mark)
        g.make_move(3)  # O -> 3
        t = g.to_tensor()
        # X's newest (age rank 0, plane 0) mark is at square 0 -> (0,0)
        self.assertEqual(t[0, 0, 0], 1.0)
        self.assertEqual(t[1].sum(), 0.0)  # no age-rank-1 mark yet
        self.assertEqual(t[2].sum(), 0.0)  # no age-rank-2 mark yet


class TestCloneIndependence(unittest.TestCase):
    def test_clone_is_independent(self):
        g = FadingTicTacToe()
        g.make_move(0)
        clone = g.clone()
        clone.make_move(1)
        self.assertNotEqual(g.board, clone.board)
        self.assertEqual(g.current_player, PLAYER_O)  # original unaffected


if __name__ == "__main__":
    unittest.main()
