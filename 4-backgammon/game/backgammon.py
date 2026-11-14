from dataclasses import dataclass
import random
import copy


WHITE = 1
BLACK = -1

BAR = -1
OFF = 24


@dataclass(frozen=True)
class Move:
    source: int
    destination: int
    die: int
    die_index: int = 0


class Backgammon:

    def __init__(self):

        self.points = [
            2, 0, 0, 0, 0, -5,
            0, -3, 0, 0, 0, 5,
            -5, 0, 0, 0, 3, 0,
            5, 0, 0, 0, 0, -2
        ]

        self.bar = {
            WHITE: 0,
            BLACK: 0
        }

        self.borne_off = {
            WHITE: 0,
            BLACK: 0
        }

        self.current_player = WHITE

        self.dice = []

        self.done = False
        self.winner = None

    # =====================================================
    # Clone
    # =====================================================

    def clone(self):
        return copy.deepcopy(self)

    # =====================================================
    # Helpers
    # =====================================================

    def opponent(self, player):
        return -player

    # =====================================================
    # Dice
    # =====================================================

    def roll_dice(self):

        if self.done:
            return []

        d1 = random.randint(1, 6)
        d2 = random.randint(1, 6)

        if d1 == d2:
            self.dice = [d1, d1, d1, d1]
        else:
            self.dice = sorted([d1, d2])

        return list(self.dice)

    def set_dice(self, dice):
        dice = list(dice)

        if len(dice) == 2:
            dice = sorted(dice)

        self.dice = dice

    # =====================================================
    # Destination
    # =====================================================

    def destination_for_die(self, source, die, player):

        if player == WHITE:
            return source + die

        return source - die

    # =====================================================
    # Block
    # =====================================================

    def is_blocked(self, destination, player):

        if destination < 0 or destination >= 24:
            return False

        opponent = self.opponent(player)

        return self.points[destination] * opponent >= 2

    # =====================================================
    # Home
    # =====================================================

    def all_checkers_home(self, player):

        if self.bar[player] > 0:
            return False

        if player == WHITE:

            for i in range(18):
                if self.points[i] > 0:
                    return False

        else:

            for i in range(6, 24):
                if self.points[i] < 0:
                    return False

        return True

    # =====================================================
    # Bear off
    # =====================================================

    def can_bear_off(self, source, die, player):

        if not self.all_checkers_home(player):
            return False

        if player == WHITE:

            distance = 24 - source

            if die == distance:
                return True

            if die > distance:

                for i in range(18, source):
                    if self.points[i] > 0:
                        return False

                return True

            return False

        distance = source + 1

        if die == distance:
            return True

        if die > distance:

            for i in range(source + 1, 6):
                if self.points[i] < 0:
                    return False

            return True

        return False

    # =====================================================
    # Legal moves for one die
    # =====================================================

    def legal_moves_for_die(
        self,
        die,
        player=None,
        die_index=0
    ):

        if player is None:
            player = self.current_player

        if self.done:
            return []

        if die < 1 or die > 6:
            return []

        moves = []

        # -------------------------------------------------
        # Bar priority
        # -------------------------------------------------

        if self.bar[player] > 0:

            if player == WHITE:
                destination = die - 1
            else:
                destination = 24 - die

            if not self.is_blocked(destination, player):

                moves.append(
                    Move(
                        source=BAR,
                        destination=destination,
                        die=die,
                        die_index=die_index
                    )
                )

            return moves

        # -------------------------------------------------
        # Normal moves
        # -------------------------------------------------

        for source in range(24):

            if self.points[source] * player <= 0:
                continue

            destination = self.destination_for_die(
                source,
                die,
                player
            )

            if 0 <= destination < 24:

                if not self.is_blocked(destination, player):

                    moves.append(
                        Move(
                            source=source,
                            destination=destination,
                            die=die,
                            die_index=die_index
                        )
                    )

            elif self.can_bear_off(source, die, player):

                moves.append(
                    Move(
                        source=source,
                        destination=OFF,
                        die=die,
                        die_index=die_index
                    )
                )

        return moves

    # =====================================================
    # All legal moves
    # =====================================================

    def legal_moves(self, player=None):

        if player is None:
            player = self.current_player

        if self.done or not self.dice:
            return []

        moves = []

        for die_index, die in enumerate(self.dice):

            moves.extend(
                self.legal_moves_for_die(
                    die=die,
                    player=player,
                    die_index=die_index
                )
            )

        return moves

    # =====================================================
    # Must pass
    # =====================================================

    def must_pass(self):

        if self.done:
            return True

        if not self.dice:
            return True

        return len(self.legal_moves()) == 0

    # =====================================================
    # Consume die
    # =====================================================

    def _consume_die(self, die_index):

        if die_index < 0 or die_index >= len(self.dice):
            raise ValueError("Invalid die index.")

        self.dice.pop(die_index)

    # =====================================================
    # Make move
    # =====================================================

    def make_move(self, move):

        if self.done:
            raise ValueError("Game is already over.")

        player = self.current_player
        opponent = self.opponent(player)

        if move.die_index < 0:
            raise ValueError("Invalid die index.")

        if move.die_index >= len(self.dice):
            raise ValueError("Invalid die index.")

        if self.dice[move.die_index] != move.die:
            raise ValueError("Die mismatch.")

        legal = self.legal_moves_for_die(
            die=move.die,
            player=player,
            die_index=move.die_index
        )

        valid = any(
            m.source == move.source
            and m.destination == move.destination
            for m in legal
        )

        if not valid:
            raise ValueError(f"Illegal move: {move}")

        self._consume_die(move.die_index)

        # Remove source checker

        if move.source == BAR:

            if self.bar[player] <= 0:
                raise RuntimeError("No checker on bar.")

            self.bar[player] -= 1

        else:

            if self.points[move.source] * player <= 0:
                raise RuntimeError(
                    "Source does not contain player's checker."
                )

            self.points[move.source] -= player

        # Bear off

        if move.destination == OFF:

            self.borne_off[player] += 1

        else:

            destination_value = self.points[move.destination]

            # Hit

            if destination_value * opponent == 1:

                self.points[move.destination] = 0
                self.bar[opponent] += 1

            self.points[move.destination] += player

        # Win

        if self.borne_off[player] >= 15:

            self.borne_off[player] = 15

            self.done = True
            self.winner = player

            self.dice = []

    # =====================================================
    # End turn
    # =====================================================

    def end_turn(self):

        if self.done:
            return

        self.dice = []

        self.current_player = self.opponent(
            self.current_player
        )

    # =====================================================
    # Encode
    # =====================================================

    def encode(self, player=None):

        if player is None:
            player = self.current_player

        opponent = self.opponent(player)

        state = []

        # =====================================================
        # Player bar
        # 0
        # =====================================================

        state.append(
            float(self.bar[player])
        )

        # =====================================================
        # Player pieces
        # 1..24
        #
        # Always encode from player's perspective.
        #
        # WHITE:
        #   points 0 -> 23
        #
        # BLACK:
        #   points 23 -> 0
        # =====================================================

        if player == WHITE:

            player_points = self.points

        else:

            player_points = reversed(self.points)

        for point in player_points:

            state.append(
                float(max(point * player, 0))
            )

        # =====================================================
        # Player bear off
        # 25
        # =====================================================

        state.append(
            float(self.borne_off[player])
        )

        # =====================================================
        # Opponent bar
        # 26
        # =====================================================

        state.append(
            float(self.bar[opponent])
        )

        # =====================================================
        # Opponent pieces
        # 27..50
        #
        # IMPORTANT:
        # Use the SAME board orientation as the player.
        # =====================================================

        if player == WHITE:

            opponent_points = self.points

        else:

            opponent_points = reversed(self.points)

        for point in opponent_points:

            state.append(
                float(max(point * opponent, 0))
            )

        # =====================================================
        # Opponent bear off
        # 51
        # =====================================================

        state.append(
            float(self.borne_off[opponent])
        )

        # =====================================================
        # Dice
        # 52..55
        # =====================================================

        dice = list(self.dice)

        while len(dice) < 4:
            dice.append(0)

        for die in dice[:4]:

            state.append(
                float(die)
            )

        return state