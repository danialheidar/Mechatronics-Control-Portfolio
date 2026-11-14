import math
import random

import torch

from game.action_space import ActionSpace
from mcts.node import Node


class MCTS:

    def __init__(
        self,
        network,
        simulations=800,
        c_puct=1.5,
        device="cpu"
    ):
        self.network = network
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device

        self.network.eval()

    # =====================================================
    # Dice utilities
    # =====================================================

    @staticmethod
    def dice_key(dice):
        """
        Canonical representation of a dice outcome.

        Doubles:
            (d, d, d, d)

        Non-doubles:
            sorted pair
        """

        dice = tuple(dice)

        if len(dice) == 4:

            if not (
                dice[0]
                == dice[1]
                == dice[2]
                == dice[3]
            ):
                raise ValueError(
                    f"Invalid double dice: {dice}"
                )

            return dice

        if len(dice) == 2:

            d1, d2 = dice

            if not (
                1 <= d1 <= 6
                and 1 <= d2 <= 6
                and d1 != d2
            ):
                raise ValueError(
                    f"Invalid non-double dice: {dice}"
                )

            return tuple(sorted(dice))

        raise ValueError(
            f"Invalid dice: {dice}"
        )

    # =====================================================
    # Network evaluation
    # =====================================================

    def evaluate(self, game):
        """
        Encode state from CURRENT PLAYER perspective.

        Action space uses exactly the same perspective.
        """

        state = game.encode(
            game.current_player
        )

        if len(state) != 56:
            raise RuntimeError(
                f"Expected state size 56, "
                f"got {len(state)}"
            )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            policy_logits, value = self.network(
                state_tensor
            )

        policy_logits = policy_logits[0]

        value = float(
            value.item()
        )

        return policy_logits, value

    # =====================================================
    # Expand Player Node
    # =====================================================

    def expand_player(self, node, game):
        """
        Expand all legal player-relative actions.
        """

        legal_actions = (
            ActionSpace.legal_action_ids(game)
        )

        if not legal_actions:
            return 0.0

        policy_logits, value = self.evaluate(game)

        legal_indices = torch.tensor(
            legal_actions,
            dtype=torch.long,
            device=self.device
        )

        legal_logits = policy_logits[
            legal_indices
        ]

        priors = torch.softmax(
            legal_logits,
            dim=0
        )

        player = game.current_player

        for i, action_id in enumerate(
            legal_actions
        ):

            node.children[action_id] = Node(
                node_type=Node.PLAYER,
                parent=node,
                prior=float(
                    priors[i].item()
                ),
                player=player
            )

        return value

    # =====================================================
    # Expand Chance Node
    # =====================================================

    def expand_chance(self, node):
        """
        Create all 21 unordered dice outcomes.

        Doubles:
            1 / 36

        Non-doubles:
            2 / 36
        """

        if node.children:
            return

        for d1 in range(1, 7):

            for d2 in range(d1, 7):

                if d1 == d2:

                    probability = 1.0 / 36.0

                    dice = (
                        d1,
                        d1,
                        d1,
                        d1
                    )

                else:

                    probability = 2.0 / 36.0

                    dice = (
                        d1,
                        d2
                    )

                node.children[dice] = Node(
                    node_type=Node.PLAYER,
                    parent=node,
                    prior=probability
                )

    # =====================================================
    # PUCT
    # =====================================================

    def puct_score(
        self,
        parent,
        child
    ):
        """
        PUCT score.

        child.value is interpreted from child's player
        perspective.

        If players differ, flip the sign.
        """

        if (
            parent.player is not None
            and child.player is not None
            and parent.player != child.player
        ):
            q = -child.value

        else:
            q = child.value

        u = (
            self.c_puct
            * child.prior
            * math.sqrt(
                parent.visit_count + 1
            )
            / (
                1 + child.visit_count
            )
        )

        return q + u

    # =====================================================
    # Select Player Child
    # =====================================================

    def select_player_child(self, node):
        """
        Select only real action children.

        Chance transitions are stored separately in
        node.chance_child.
        """

        action_children = [
            (action_id, child)
            for action_id, child
            in node.children.items()
            if isinstance(action_id, int)
        ]

        if not action_children:
            raise RuntimeError(
                "Player node has no action children."
            )

        return max(
            action_children,
            key=lambda item: self.puct_score(
                node,
                item[1]
            )
        )

    # =====================================================
    # Select Chance Child
    # =====================================================

    def select_chance_child(self, node):

        if not node.children:
            raise RuntimeError(
                "Chance node has no outcomes."
            )

        outcomes = list(
            node.children.items()
        )

        probabilities = [
            child.prior
            for _, child in outcomes
        ]

        return random.choices(
            outcomes,
            weights=probabilities,
            k=1
        )[0]

    # =====================================================
    # Apply Action
    # =====================================================

    def apply_action(
        self,
        game,
        action_id
    ):
        """
        action_id is player-relative.

        ActionSpace converts it to the actual board source.
        """

        move = ActionSpace.action_to_move(
            game,
            action_id
        )

        if move is None:

            player = game.current_player

            try:
                source, die = ActionSpace.decode(
                    action_id,
                    player
                )
            except ValueError:
                source = None
                die = None

            raise RuntimeError(
                "\nMCTS selected an illegal action.\n"
                f"action_id = {action_id}\n"
                f"source    = {source}\n"
                f"die       = {die}\n"
                f"dice      = {game.dice}\n"
                f"player    = {player}\n"
            )

        game.make_move(move)

    # =====================================================
    # Apply Dice
    # =====================================================

    def apply_dice(
        self,
        game,
        dice
    ):

        game.set_dice(
            list(dice)
        )

    # =====================================================
    # Terminal Value
    # =====================================================

    def terminal_value(self, game):

        if not game.done:
            raise RuntimeError(
                "terminal_value() called on non-terminal game."
            )

        if game.winner == game.current_player:
            return 1.0

        return -1.0

    # =====================================================
    # Backup
    # =====================================================

    def backup(
        self,
        search_path,
        value
    ):
        """
        Backup value through the search path.

        Values are always interpreted from the perspective
        of the node's player.

        PLAYER -> same PLAYER:
            same value

        PLAYER -> opponent:
            sign flip

        PLAYER -> CHANCE:
            same player

        CHANCE -> PLAYER:
            same player
        """

        for i in reversed(
            range(len(search_path))
        ):

            node = search_path[i]

            node.visit_count += 1
            node.value_sum += value

            if i == 0:
                break

            parent = search_path[i - 1]

            if (
                parent.player is not None
                and node.player is not None
                and parent.player != node.player
            ):
                value = -value

    # =====================================================
    # Simulation
    # =====================================================

    def simulate(
        self,
        root,
        game
    ):

        node = root

        search_path = [node]

        while True:

            # =================================================
            # TERMINAL
            # =================================================

            if game.done:

                value = self.terminal_value(
                    game
                )

                self.backup(
                    search_path,
                    value
                )

                return

            # =================================================
            # CHANCE NODE
            # =================================================

            if node.node_type == Node.CHANCE:

                if node.player is None:
                    node.player = (
                        game.current_player
                    )

                if not node.is_expanded():
                    self.expand_chance(node)

                dice_key, child = (
                    self.select_chance_child(node)
                )

                self.apply_dice(
                    game,
                    dice_key
                )

                child.node_type = Node.PLAYER
                child.player = (
                    game.current_player
                )

                node = child

                search_path.append(node)

                continue

            # =================================================
            # PLAYER NODE
            # =================================================

            if node.node_type == Node.PLAYER:

                if node.player is None:
                    node.player = (
                        game.current_player
                    )

                # -------------------------------------------------
                # No legal moves
                # -------------------------------------------------

                legal_actions = (
                    ActionSpace.legal_action_ids(
                        game
                    )
                )

                if not legal_actions:

                    # No PASS action exists.
                    #
                    # Therefore the turn ends.

                    game.end_turn()

                    chance = node.chance_child

                    if chance is None:

                        chance = Node(
                            node_type=Node.CHANCE,
                            parent=node,
                            prior=1.0,
                            player=game.current_player
                        )

                        node.chance_child = chance

                    else:

                        chance.parent = node
                        chance.node_type = (
                            Node.CHANCE
                        )
                        chance.player = (
                            game.current_player
                        )

                    node = chance

                    search_path.append(node)

                    continue

                # -------------------------------------------------
                # Expand leaf
                # -------------------------------------------------

                if not node.is_expanded():

                    value = self.expand_player(
                        node,
                        game
                    )

                    self.backup(
                        search_path,
                        value
                    )

                    return

                # -------------------------------------------------
                # Select action
                # -------------------------------------------------

                action_id, child = (
                    self.select_player_child(node)
                )

                self.apply_action(
                    game,
                    action_id
                )

                # After a move, the current player normally
                # remains unchanged until dice are exhausted.

                child.player = (
                    game.current_player
                )

                node = child

                search_path.append(node)

                # -------------------------------------------------
                # Terminal after move
                # -------------------------------------------------

                if game.done:

                    value = self.terminal_value(
                        game
                    )

                    self.backup(
                        search_path,
                        value
                    )

                    return

                # -------------------------------------------------
                # More dice remain
                # -------------------------------------------------

                if game.dice:

                    node.node_type = Node.PLAYER

                    continue

                # -------------------------------------------------
                # Dice exhausted
                # -------------------------------------------------

                game.end_turn()

                chance = node.chance_child

                if chance is None:

                    chance = Node(
                        node_type=Node.CHANCE,
                        parent=node,
                        prior=1.0,
                        player=game.current_player
                    )

                    node.chance_child = chance

                else:

                    chance.parent = node
                    chance.node_type = (
                        Node.CHANCE
                    )
                    chance.player = (
                        game.current_player
                    )

                node = chance

                search_path.append(node)

                continue

            raise RuntimeError(
                f"Unknown node type: {node.node_type}"
            )

    # =====================================================
    # Search
    # =====================================================

    def search(
        self,
        game,
        root=None
    ):
        """
        Run MCTS simulations.

        Root is:
            PLAYER if dice already exist
            CHANCE if a dice roll is required
        """

        expected_type = (
            Node.PLAYER
            if game.dice
            else Node.CHANCE
        )

        if root is not None:

            if root.node_type != expected_type:
                root = None

            elif root.player != game.current_player:
                root = None

        if root is None:

            root = Node(
                node_type=expected_type,
                player=game.current_player,
                state=game.encode(
                    game.current_player
                )
            )

        for _ in range(self.simulations):

            simulation_game = game.clone()

            self.simulate(
                root,
                simulation_game
            )

        return root

    # =====================================================
    # Visit Policy
    # =====================================================

    def visit_policy(
        self,
        root
    ):
        """
        Return AlphaZero-style visit-count policy.

        Policy indices are the 150 player-relative
        action IDs.
        """

        policy = [
            0.0
        ] * ActionSpace.SIZE

        if root.node_type != Node.PLAYER:
            return policy

        action_children = [
            (action, child)
            for action, child
            in root.children.items()
            if isinstance(action, int)
        ]

        total = sum(
            child.visit_count
            for _, child
            in action_children
        )

        if total == 0:
            return policy

        for action_id, child in action_children:

            policy[action_id] = (
                child.visit_count
                / total
            )

        return policy

    # =====================================================
    # Select Action
    # =====================================================

    def select_action(
        self,
        root,
        temperature=1.0
    ):
        """
        Select a player-relative action ID.
        """

        if root.node_type != Node.PLAYER:
            raise RuntimeError(
                "Cannot select an action from a "
                "non-player root."
            )

        actions = [
            action
            for action in root.children.keys()
            if isinstance(action, int)
        ]

        if not actions:
            raise RuntimeError(
                "No actions available."
            )

        visits = [
            root.children[action].visit_count
            for action in actions
        ]

        # -------------------------------------------------
        # Greedy
        # -------------------------------------------------

        if temperature <= 0:

            return actions[
                max(
                    range(len(actions)),
                    key=lambda i: visits[i]
                )
            ]

        # -------------------------------------------------
        # Temperature sampling
        # -------------------------------------------------

        exponent = 1.0 / temperature

        weights = [
            max(v, 0) ** exponent
            for v in visits
        ]

        if sum(weights) <= 0:

            weights = [
                1.0
                for _ in actions
            ]

        return random.choices(
            actions,
            weights=weights,
            k=1
        )[0]

    # =====================================================
    # Reuse Action Subtree
    # =====================================================

    def reuse_action_root(
        self,
        root,
        action_id
    ):
        """
        After executing an action in the real game,
        reuse the corresponding child as the new root.
        """

        if root is None:
            return None

        child = root.children.get(
            action_id
        )

        if child is None:
            return None

        child.parent = None

        return child

    # =====================================================
    # Reuse Chance Subtree
    # =====================================================

    def reuse_chance_root(
        self,
        root,
        dice
    ):
        """
        After a real dice roll, reuse the corresponding
        chance child.
        """

        if root is None:
            return None

        key = self.dice_key(
            dice
        )

        child = root.children.get(
            key
        )

        if child is None:
            return None

        child.parent = None

        return child