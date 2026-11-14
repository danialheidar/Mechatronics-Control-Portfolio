from game.backgammon import BAR, WHITE, BLACK


class ActionSpace:
    """
    Player-relative action space for Backgammon.

    24 board sources + BAR
    6 possible die values

    Total:
        25 * 6 = 150 actions

    IMPORTANT:
    Action source is encoded from the CURRENT PLAYER'S perspective.

    WHITE:
        relative 0 -> actual point 0
        relative 23 -> actual point 23

    BLACK:
        relative 0 -> actual point 23
        relative 23 -> actual point 0

    BAR:
        relative 24 -> BAR
    """

    # -----------------------------------------------------
    # Constants
    # -----------------------------------------------------

    NUM_BOARD_POINTS = 24
    NUM_SOURCES = 25

    NUM_DICE = 6

    BAR_SOURCE = 24

    SIZE = NUM_SOURCES * NUM_DICE

    # -----------------------------------------------------
    # Player validation
    # -----------------------------------------------------

    @staticmethod
    def _validate_player(player):
        if player not in (WHITE, BLACK):
            raise ValueError(
                f"Invalid player: {player}"
            )

    # -----------------------------------------------------
    # Source conversion
    # -----------------------------------------------------

    @staticmethod
    def relative_source(actual_source, player):
        """
        Convert actual board source -> player-relative source.

        WHITE:
            actual 0  -> relative 0
            actual 23 -> relative 23

        BLACK:
            actual 23 -> relative 0
            actual 0  -> relative 23

        BAR remains 24.
        """

        ActionSpace._validate_player(player)

        if actual_source == BAR:
            return ActionSpace.BAR_SOURCE

        if not 0 <= actual_source < 24:
            raise ValueError(
                f"Invalid actual source: {actual_source}"
            )

        if player == WHITE:
            return actual_source

        return 23 - actual_source

    # -----------------------------------------------------

    @staticmethod
    def actual_source(relative_source, player):
        """
        Convert player-relative source -> actual board source.

        BAR remains BAR.
        """

        ActionSpace._validate_player(player)

        if relative_source == ActionSpace.BAR_SOURCE:
            return BAR

        if not 0 <= relative_source < 24:
            raise ValueError(
                f"Invalid relative source: {relative_source}"
            )

        if player == WHITE:
            return relative_source

        return 23 - relative_source

    # -----------------------------------------------------
    # Encode
    # -----------------------------------------------------

    @staticmethod
    def encode(source, die, player):
        """
        Encode an ACTUAL board source and die into an
        action ID using player-relative orientation.

        Example:

            WHITE:
                source=0, die=1 -> action 0

            BLACK:
                source=23, die=1 -> action 0

            BAR:
                die=1 -> action 144
        """

        ActionSpace._validate_player(player)

        if source == BAR:
            relative_source = ActionSpace.BAR_SOURCE

        else:
            relative_source = ActionSpace.relative_source(
                source,
                player
            )

        if not 0 <= relative_source < ActionSpace.NUM_SOURCES:
            raise ValueError(
                f"Invalid source: {source}"
            )

        if not 1 <= die <= 6:
            raise ValueError(
                f"Invalid die: {die}"
            )

        return (
            relative_source * ActionSpace.NUM_DICE
            + (die - 1)
        )

    # -----------------------------------------------------
    # Decode
    # -----------------------------------------------------

    @staticmethod
    def decode(action_id, player):
        """
        Decode action ID into an ACTUAL board source and die.

        The returned source is always in the actual game
        board coordinate system.
        """

        ActionSpace._validate_player(player)

        if not 0 <= action_id < ActionSpace.SIZE:
            raise ValueError(
                f"Invalid action id: {action_id}"
            )

        relative_source = (
            action_id // ActionSpace.NUM_DICE
        )

        die = (
            action_id % ActionSpace.NUM_DICE
        ) + 1

        source = ActionSpace.actual_source(
            relative_source,
            player
        )

        return source, die

    # -----------------------------------------------------
    # Legal action IDs
    # -----------------------------------------------------

    @staticmethod
    def legal_action_ids(game):
        """
        Return unique player-relative action IDs for all
        legal moves in the current game state.
        """

        if game.done:
            return []

        if not game.dice:
            return []

        player = game.current_player

        actions = set()

        for move in game.legal_moves():

            action_id = ActionSpace.encode(
                source=move.source,
                die=move.die,
                player=player
            )

            actions.add(action_id)

        return sorted(actions)

    # -----------------------------------------------------
    # Action -> Move
    # -----------------------------------------------------

    @staticmethod
    def action_to_move(game, action_id):
        """
        Convert a player-relative action ID into the actual
        legal Move object for the CURRENT game state.

        This is the safest way to execute an action because
        the returned Move contains the correct die_index.
        """

        if game.done:
            return None

        player = game.current_player

        source, die = ActionSpace.decode(
            action_id,
            player
        )

        for move in game.legal_moves():

            if (
                move.source == source
                and move.die == die
            ):
                return move

        return None