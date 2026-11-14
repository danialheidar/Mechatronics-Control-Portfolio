import os
import glob
import re
import time
import random

import torch

from game.backgammon import Backgammon, WHITE, BLACK
from game.action_space import ActionSpace
from mcts.mcts import MCTS
from mcts.node import Node
from network.model import AlphaZeroNetwork
from renderer import BackgammonRenderer


# =========================================================
# Configuration
# =========================================================

CHECKPOINT_DIR = "checkpoints"

# All AlphaZero checkpoints
CHECKPOINT_PATTERN = os.path.join(
    CHECKPOINT_DIR,
    "*.pt"
)

# MCTS simulations used for each move
SIMULATIONS = 3000

# PUCT exploration constant
C_PUCT = 1.5

# Temperature for move selection
#
# 0.0 = completely greedy
# 1.0 = sample according to visit counts
#
# For evaluation we normally want 0.0.
TEMPERATURE = 0.0

# Delay between moves
RENDER_DELAY = 0.0

# Maximum number of turns
MAX_TURNS = 2000

# Time to show final board
END_DELAY = 5.0


# =========================================================
# Network configuration
# =========================================================

STATE_SIZE = 56

# 25 * 6
ACTION_SIZE = 150


# =========================================================
# Find latest checkpoints
# =========================================================

def extract_checkpoint_number(path):

    filename = os.path.basename(path)

    # Examples:
    #
    # checkpoint_100.pt
    # checkpoint_5000.pt
    # alphazero_25000.pt
    #
    match = re.search(
        r"(\d+)(?=\.pt$)",
        filename
    )

    if match is None:
        return None

    return int(match.group(1))


def find_latest_checkpoints():

    checkpoints = glob.glob(
        CHECKPOINT_PATTERN
    )

    if len(checkpoints) < 2:

        raise RuntimeError(
            f"At least 2 checkpoints are required.\n"
            f"Found: {len(checkpoints)}\n"
            f"Directory: {CHECKPOINT_DIR}"
        )

    # -----------------------------------------------------
    # Prefer numeric checkpoint number
    # -----------------------------------------------------

    numbered = []

    for path in checkpoints:

        number = extract_checkpoint_number(path)

        if number is not None:

            numbered.append(
                (number, path)
            )

    if len(numbered) >= 2:

        numbered.sort(
            key=lambda x: x[0]
        )

        return (
            numbered[-2][1],
            numbered[-1][1]
        )

    # -----------------------------------------------------
    # Fallback to modification time
    # -----------------------------------------------------

    checkpoints.sort(
        key=os.path.getmtime
    )

    return (
        checkpoints[-2],
        checkpoints[-1]
    )


# =========================================================
# Load network
# =========================================================

def create_network():

    # -----------------------------------------------------
    # IMPORTANT
    #
    # This assumes your network class is called
    # AlphaZeroNet and accepts:
    #
    # input_size
    # action_size
    #
    # If your model.py uses another constructor,
    # change this function only.
    # -----------------------------------------------------

    return AlphaZeroNetwork(
        input_size=STATE_SIZE,
        action_size=ACTION_SIZE
    )


def load_network(checkpoint_path, device):

    network = create_network()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # -----------------------------------------------------
    # Different common checkpoint formats
    # -----------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        # state_dict directly
        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "network_state_dict" in checkpoint:

            state_dict = checkpoint[
                "network_state_dict"
            ]

        else:

            # Assume the dictionary itself is
            # a state_dict.
            state_dict = checkpoint

        # Remove possible DataParallel prefix
        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith(
                "module."
            ):

                key = key[
                    len("module.") :
                ]

            cleaned_state_dict[
                key
            ] = value

        network.load_state_dict(
            cleaned_state_dict
        )

    else:

        # If the whole model was saved
        if isinstance(
            checkpoint,
            torch.nn.Module
        ):

            network = checkpoint

        else:

            raise RuntimeError(
                f"Unsupported checkpoint format: "
                f"{checkpoint_path}"
            )

    network.to(device)

    network.eval()

    return network


# =========================================================
# Checkpoint information
# =========================================================

def print_match_info(
    white_checkpoint,
    black_checkpoint,
    device
):

    print()
    print(
        "=================================================="
    )
    print(
        "              ALPHAZERO BACKGAMMON"
    )
    print(
        "=================================================="
    )

    print()
    print(
        "White AI:"
    )
    print(
        f"  {white_checkpoint}"
    )

    print()
    print(
        "Black AI:"
    )
    print(
        f"  {black_checkpoint}"
    )

    print()
    print(
        f"State size:       {STATE_SIZE}"
    )

    print(
        f"Action size:      {ACTION_SIZE}"
    )

    print(
        f"MCTS simulations: {SIMULATIONS}"
    )

    print(
        f"PUCT:             {C_PUCT}"
    )

    print(
        f"Temperature:      {TEMPERATURE}"
    )

    print(
        f"Device:           {device}"
    )

    print()
    print(
        "=================================================="
    )
    print()


# =========================================================
# Create root for current game state
# =========================================================

def create_root(game):

    if game.done:
        return None

    if game.dice:

        return Node(
            node_type=Node.PLAYER,
            player=game.current_player,
            state=game.encode(
                game.current_player
            )
        )

    return Node(
        node_type=Node.CHANCE,
        player=game.current_player,
        state=game.encode(
            game.current_player
        )
    )


# =========================================================
# Roll actual dice
# =========================================================

def roll_actual_dice(
    game,
    root,
    mcts
):

    dice = tuple(
        game.roll_dice()
    )

    # -----------------------------------------------------
    # Reuse the chance subtree corresponding to the
    # actual dice outcome.
    # -----------------------------------------------------

    reused_root = mcts.reuse_chance_root(
        root,
        dice
    )

    if reused_root is not None:

        return reused_root

    # This can happen if the actual outcome was not visited
    # during MCTS simulations.
    #
    # In that case create a fresh player root.

    return Node(
        node_type=Node.PLAYER,
        player=game.current_player,
        state=game.encode(
            game.current_player
        )
    )


# =========================================================
# Play one AI turn
# =========================================================

def play_ai_turn(
    game,
    mcts,
    root,
    renderer
):

    player = game.current_player

    # -----------------------------------------------------
    # Check for a forced pass.
    #
    # The player may have dice but no legal moves.
    # In that case there is no action to select.
    # -----------------------------------------------------

    legal_moves = game.legal_moves()

    if not legal_moves:

        player_name = (
            "WHITE"
            if player == WHITE
            else "BLACK"
        )

        print(
            f"{player_name}: no legal moves -- PASS"
        )

        # No action can be played, so the turn ends.
        game.end_turn()

        # The next state is a chance node for the
        # opponent.
        return Node(
            node_type=Node.CHANCE,
            player=game.current_player,
            state=game.encode(
                game.current_player
            )
        )

    # -----------------------------------------------------
    # Search current state
    # -----------------------------------------------------

    root = mcts.search(
        game,
        root
    )

    # -----------------------------------------------------
    # Select action
    # -----------------------------------------------------

    action_id = mcts.select_action(
        root,
        temperature=TEMPERATURE
    )

    # -----------------------------------------------------
    # Decode only for logging
    #
    # Action IDs are player-relative, so the current player
    # MUST be supplied when decoding.
    # -----------------------------------------------------

    source, die = ActionSpace.decode(
        action_id,
        player
    )

    # -----------------------------------------------------
    # Apply action
    # -----------------------------------------------------

    mcts.apply_action(
        game,
        action_id
    )

    # -----------------------------------------------------
    # Render
    # -----------------------------------------------------

    renderer.render(
        game,
        delay=RENDER_DELAY
    )

    # -----------------------------------------------------
    # Print move
    # -----------------------------------------------------

    player_name = (
        "WHITE"
        if player == WHITE
        else "BLACK"
    )

    if source == -1:

        source_text = "BAR"

    else:

        source_text = str(
            source + 1
        )

    if game.done:

        destination_text = "OFF"

    else:

        # We need to infer destination from the move.
        destination_text = "..."

    print(
        f"{player_name}: "
        f"action={action_id} "
        f"source={source_text} "
        f"die={die}"
    )

    # -----------------------------------------------------
    # If game finished
    # -----------------------------------------------------

    if game.done:

        return root

    # -----------------------------------------------------
    # Dice remain:
    #
    # Same player continues.
    # Reuse selected action subtree.
    # -----------------------------------------------------

    if game.dice:

        new_root = (
            mcts.reuse_action_root(
                root,
                action_id
            )
        )

        if new_root is not None:

            return new_root

        return Node(
            node_type=Node.PLAYER,
            player=game.current_player,
            state=game.encode(
                game.current_player
            )
        )

    # -----------------------------------------------------
    # Turn finished
    # -----------------------------------------------------

    child = (
        mcts.reuse_action_root(
            root,
            action_id
        )
    )

    # Change player
    game.end_turn()

    # -----------------------------------------------------
    # Try to reuse the chance node that MCTS already built.
    # -----------------------------------------------------

    if child is not None:

        chance = child.chance_child

        if chance is not None:

            chance.parent = None

            chance.node_type = Node.CHANCE

            chance.player = (
                game.current_player
            )

            return chance

    # -----------------------------------------------------
    # No existing chance subtree.
    # -----------------------------------------------------

    return Node(
        node_type=Node.CHANCE,
        player=game.current_player
    )


# =========================================================
# Main
# =========================================================

def main():

    # =====================================================
    # Check checkpoint directory
    # =====================================================

    if not os.path.exists(
        CHECKPOINT_DIR
    ):

        print(
            f"Checkpoint directory does not exist: "
            f"{CHECKPOINT_DIR}"
        )

        return

    # =====================================================
    # Find two latest checkpoints
    # =====================================================

    try:

        checkpoint_a, checkpoint_b = (
            find_latest_checkpoints()
        )

    except RuntimeError as e:

        print(e)

        return

    # =====================================================
    # Device
    # =====================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print_match_info(
        checkpoint_a,
        checkpoint_b,
        device
    )

    # =====================================================
    # Load networks
    # =====================================================

    print(
        "Loading White AI..."
    )

    network_white = load_network(
        checkpoint_a,
        device
    )

    print(
        "Loading Black AI..."
    )

    network_black = load_network(
        checkpoint_b,
        device
    )

    print(
        "Networks loaded."
    )

    # =====================================================
    # Create MCTS
    # =====================================================

    mcts_white = MCTS(
        network=network_white,
        simulations=SIMULATIONS,
        c_puct=C_PUCT,
        device=device
    )

    mcts_black = MCTS(
        network=network_black,
        simulations=SIMULATIONS,
        c_puct=C_PUCT,
        device=device
    )

    # =====================================================
    # Game
    # =====================================================

    game = Backgammon()

    # =====================================================
    # Renderer
    # =====================================================

    renderer = BackgammonRenderer(
        enabled=True
    )

    # =====================================================
    # Roots
    # =====================================================

    white_root = None
    black_root = None

    # =====================================================
    # Statistics
    # =====================================================

    turns = 0

    white_moves = 0
    black_moves = 0

    # =====================================================
    # Match
    # =====================================================

    try:

        while (
            not game.done
            and
            turns < MAX_TURNS
            and
            renderer.running
        ):

            # -------------------------------------------------
            # Current player
            # -------------------------------------------------

            player = (
                game.current_player
            )

            # =================================================
            # Chance phase
            # =================================================

            if not game.dice:

                if player == WHITE:

                    mcts = mcts_white

                    root = white_root

                else:

                    mcts = mcts_black

                    root = black_root

                # -------------------------------------------------
                # If no root exists, create chance root
                # -------------------------------------------------

                if root is None:

                    root = Node(
                        node_type=Node.CHANCE,
                        player=player
                    )

                # -------------------------------------------------
                # MCTS searches the chance node
                # -------------------------------------------------

                root = mcts.search(
                    game,
                    root
                )

                # -------------------------------------------------
                # Actual dice roll
                # -------------------------------------------------

                dice = tuple(
                    game.roll_dice()
                )

                print()
                print(
                    "------------------------------------------"
                )
                print(
                    f"{'WHITE' if player == WHITE else 'BLACK'} "
                    f"dice: {dice}"
                )
                print(
                    "------------------------------------------"
                )

                # -------------------------------------------------
                # Reuse chance subtree
                # -------------------------------------------------

                root = (
                    mcts.reuse_chance_root(
                        root,
                        dice
                    )
                )

                # -------------------------------------------------
                # If outcome wasn't expanded
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
                # Store root
                # -------------------------------------------------

                if player == WHITE:

                    white_root = root

                else:

                    black_root = root

                # Don't increment turn here.
                continue

            # =================================================
            # Player phase
            # =================================================

            if player == WHITE:

                mcts = mcts_white
                root = white_root

            else:

                mcts = mcts_black
                root = black_root

            # -------------------------------------------------
            # Play one move
            # -------------------------------------------------

            before_dice = list(
                game.dice
            )

            root = play_ai_turn(
                game,
                mcts,
                root,
                renderer
            )

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------

            if player == WHITE:

                white_moves += 1

            else:

                black_moves += 1

            turns += 1

            # -------------------------------------------------
            # Save root
            # -------------------------------------------------

            if player == WHITE:

                white_root = root

            else:

                black_root = root

            # -------------------------------------------------
            # Terminal
            # -------------------------------------------------

            if game.done:
                break

            # -------------------------------------------------
            # If dice still remain:
            #
            # same player continues.
            # -------------------------------------------------

            if game.dice:

                continue

            # -------------------------------------------------
            # Turn has already been ended inside
            # play_ai_turn().
            #
            # Therefore next iteration starts at
            # chance phase.
            # -------------------------------------------------

        # =====================================================
        # Final render
        # =====================================================

        if renderer.running:

            renderer.render(
                game,
                delay=0.0
            )

        # =====================================================
        # Result
        # =====================================================

        print()
        print(
            "=================================================="
        )

        print(
            "                  GAME FINISHED"
        )

        print(
            "=================================================="
        )

        if game.done:

            if game.winner == WHITE:

                winner_name = "WHITE"

            else:

                winner_name = "BLACK"

            print(
                f"WINNER: {winner_name}"
            )

            if game.winner == WHITE:

                print(
                    f"White AI: {checkpoint_a}"
                )

            else:

                print(
                    f"Black AI: {checkpoint_b}"
                )

        else:

            print(
                "Game stopped before completion."
            )

            print(
                f"Maximum turns: {MAX_TURNS}"
            )

        print()

        print(
            f"White checkpoint: {checkpoint_a}"
        )

        print(
            f"Black checkpoint: {checkpoint_b}"
        )

        print()

        print(
            f"White moves: {white_moves}"
        )

        print(
            f"Black moves: {black_moves}"
        )

        print(
            f"Turns: {turns}"
        )

        print()

        print(
            f"White borne off: "
            f"{game.borne_off[WHITE]}"
        )

        print(
            f"Black borne off: "
            f"{game.borne_off[BLACK]}"
        )

        print(
            "=================================================="
        )

        # =====================================================
        # Keep final board visible
        # =====================================================

        if renderer.running:

            end_time = (
                time.time()
                +
                END_DELAY
            )

            while (
                time.time() < end_time
                and
                renderer.running
            ):

                renderer.handle_events()

                renderer.clock.tick(
                    renderer.fps
                )

    except KeyboardInterrupt:

        print()
        print(
            "Game interrupted."
        )

    finally:

        renderer.close()


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    main()