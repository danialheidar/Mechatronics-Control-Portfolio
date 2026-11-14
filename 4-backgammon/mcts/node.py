class Node:

    PLAYER = "player"
    CHANCE = "chance"

    def __init__(
        self,
        node_type,
        parent=None,
        prior=0.0,
        player=None,
        state=None,
    ):
        if node_type not in (
            Node.PLAYER,
            Node.CHANCE
        ):
            raise ValueError(
                f"Invalid node type: {node_type}"
            )

        self.node_type = node_type
        self.parent = parent
        self.prior = float(prior)

        # Player whose perspective this node's value
        # belongs to.
        #
        # PLAYER node:
        #     player who is making the decision.
        #
        # CHANCE node:
        #     player who will receive the dice.
        self.player = player

        self.state = state

        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0

        # PLAYER:
        #     action_id -> child
        #
        # CHANCE:
        #     dice_tuple -> child
        self.children = {}

        # PLAYER -> CHANCE transition after the turn ends.
        self.chance_child = None

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0

        return (
            self.value_sum
            / self.visit_count
        )

    def is_expanded(self):

        if self.node_type == Node.PLAYER:

            return any(
                isinstance(key, int)
                for key in self.children
            )

        if self.node_type == Node.CHANCE:

            return len(self.children) > 0

        return False

    def __repr__(self):

        if self.node_type == Node.PLAYER:

            child_count = sum(
                1
                for key in self.children
                if isinstance(key, int)
            )

        else:

            child_count = len(
                self.children
            )

        return (
            f"Node("
            f"type={self.node_type}, "
            f"player={self.player}, "
            f"visits={self.visit_count}, "
            f"value={self.value:.4f}, "
            f"children={child_count}"
            f")"
        )