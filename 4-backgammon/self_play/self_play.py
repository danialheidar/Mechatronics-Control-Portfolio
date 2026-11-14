from dataclasses import dataclass

from game.backgammon import Backgammon
from mcts.node import Node


@dataclass
class TrainingExample:

    state: list
    policy: list
    player: int


class SelfPlay:

    def __init__(
        self,
        mcts,
        temperature=1.0
    ):

        self.mcts = mcts
        self.temperature = temperature

    # =====================================================
    # Play one complete game
    # =====================================================

    def play_game(self):

        game = Backgammon()

        # MCTS root.
        #
        # At the beginning of a turn:
        #
        #     no dice -> CHANCE
        #     dice    -> PLAYER
        #
        root = None

        history = []

        while not game.done:

            # =================================================
            # CHANCE
            # =================================================
            #
            # We need to roll dice.
            # The chance tree contains the 21 possible
            # unordered dice outcomes.
            # =================================================

            if not game.dice:

                # -------------------------------------------------
                # Create chance root if necessary
                # -------------------------------------------------

                if root is None:

                    root = Node(
                        node_type=Node.CHANCE,
                        player=game.current_player,
                        state=game.encode(
                            game.current_player
                        )
                    )

                # -------------------------------------------------
                # Search chance node
                # -------------------------------------------------

                root = self.mcts.search(
                    game,
                    root
                )

                # -------------------------------------------------
                # Actual environment dice roll
                # -------------------------------------------------

                dice = tuple(
                    game.roll_dice()
                )

                # -------------------------------------------------
                # Reuse the subtree corresponding to the
                # actual dice outcome.
                # -------------------------------------------------

                next_root = (
                    self.mcts.reuse_chance_root(
                        root,
                        dice
                    )
                )

                if next_root is None:

                    # This can happen if the actual dice outcome
                    # was not visited during the limited number
                    # of MCTS simulations.
                    #
                    # We can safely create a fresh PLAYER root.

                    root = Node(
                        node_type=Node.PLAYER,
                        player=game.current_player,
                        state=game.encode(
                            game.current_player
                        )
                    )

                else:

                    root = next_root

                    root.node_type = Node.PLAYER
                    root.player = game.current_player
                    root.state = game.encode(
                        game.current_player
                    )

                continue

            # =================================================
            # PLAYER
            # =================================================

            # -------------------------------------------------
            # IMPORTANT:
            #
            # If the player has no legal move, there is no PASS
            # action in our 150-action space.
            #
            # Therefore we must end the turn BEFORE calling
            # MCTS/select_action.
            # -------------------------------------------------

            legal_moves = game.legal_moves()

            if not legal_moves:

                # No training example is created because there
                # is no actual player action.

                game.end_turn()

                # -------------------------------------------------
                # Move from PLAYER to CHANCE.
                # -------------------------------------------------

                if root is not None:

                    chance = root.chance_child

                else:

                    chance = None

                if chance is None:

                    chance = Node(
                        node_type=Node.CHANCE,
                        parent=None,
                        prior=1.0,
                        player=game.current_player,
                        state=game.encode(
                            game.current_player
                        )
                    )

                else:

                    chance.parent = None
                    chance.node_type = Node.CHANCE
                    chance.player = game.current_player
                    chance.state = game.encode(
                        game.current_player
                    )

                root = chance

                continue

            # -------------------------------------------------
            # Make sure root is a PLAYER node
            # -------------------------------------------------

            if root is None:

                root = Node(
                    node_type=Node.PLAYER,
                    player=game.current_player,
                    state=game.encode(
                        game.current_player
                    )
                )

            # -------------------------------------------------
            # Search PLAYER node
            # -------------------------------------------------

            root = self.mcts.search(
                game,
                root
            )

            # -------------------------------------------------
            # Safety check
            # -------------------------------------------------

            if root.node_type != Node.PLAYER:

                raise RuntimeError(
                    "SelfPlay expected PLAYER root, "
                    f"got {root.node_type}"
                )

            # -------------------------------------------------
            # Get MCTS policy
            # -------------------------------------------------

            policy = self.mcts.visit_policy(
                root
            )

            # -------------------------------------------------
            # Store state BEFORE action
            # -------------------------------------------------

            state = game.encode(
                game.current_player
            )

            player = game.current_player

            history.append(
                TrainingExample(
                    state=state,
                    policy=policy,
                    player=player
                )
            )

            # -------------------------------------------------
            # Select action
            # -------------------------------------------------

            action_id = self.mcts.select_action(
                root,
                temperature=self.temperature
            )

            # -------------------------------------------------
            # Apply action
            # -------------------------------------------------
            #
            # MCTS.apply_action() itself calls game.make_move().
            # Therefore we must NOT call game.make_move() again.
            # -------------------------------------------------

            self.mcts.apply_action(
                game,
                action_id
            )

            # =================================================
            # TERMINAL
            # =================================================

            if game.done:
                break

            # =================================================
            # DICE REMAIN
            # =================================================
            #
            # Same player continues the turn.
            # Reuse the selected action subtree.
            # =================================================

            if game.dice:

                next_root = (
                    self.mcts.reuse_action_root(
                        root,
                        action_id
                    )
                )

                if next_root is None:

                    root = Node(
                        node_type=Node.PLAYER,
                        player=game.current_player,
                        state=game.encode(
                            game.current_player
                        )
                    )

                else:

                    root = next_root

                    root.node_type = Node.PLAYER
                    root.player = game.current_player
                    root.state = game.encode(
                        game.current_player
                    )

                continue

            # =================================================
            # DICE EXHAUSTED
            # =================================================
            #
            # The current player has completed the turn.
            #
            # IMPORTANT:
            # apply_action() does NOT call end_turn().
            # Therefore SelfPlay calls it exactly once here.
            # =================================================

            child = (
                self.mcts.reuse_action_root(
                    root,
                    action_id
                )
            )

            # End current player's turn.
            game.end_turn()

            # -------------------------------------------------
            # Move to CHANCE node.
            # -------------------------------------------------

            if child is not None:

                chance = child.chance_child

            else:

                chance = None

            if chance is None:

                chance = Node(
                    node_type=Node.CHANCE,
                    parent=None,
                    prior=1.0,
                    player=game.current_player,
                    state=game.encode(
                        game.current_player
                    )
                )

            else:

                chance.parent = None
                chance.node_type = Node.CHANCE
                chance.player = game.current_player
                chance.state = game.encode(
                    game.current_player
                )

            root = chance

        # =====================================================
        # Game finished
        # =====================================================

        return self.make_examples(
            history,
            game
        )

    # =====================================================
    # Assign final result
    # =====================================================

    def make_examples(
        self,
        history,
        game
    ):

        result = []

        for example in history:

            if example.player == game.winner:
                value = 1.0
            else:
                value = -1.0

            result.append(
                (
                    example.state,
                    example.policy,
                    value
                )
            )

        return result