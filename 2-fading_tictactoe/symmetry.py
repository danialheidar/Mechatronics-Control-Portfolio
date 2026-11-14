"""
symmetry.py
===========

The 3x3 board has the 8 symmetries of the square (the dihedral group
D4): 4 rotations x {identity, mirror}. Because our state encoding
(see ``game.FadingTicTacToe.to_tensor``) represents every piece of
information -- including move age -- as a *spatial* plane over the
3x3 grid (which square holds "X's newest mark", "X's oldest mark",
etc.), applying a D4 transform to the spatial (row, col) axes of every
plane is always a valid symmetry of the game: it relabels squares
consistently everywhere the square index appears, and never touches
the *channel* axis (age rank), which is exactly where "which mark is
about to fade" is encoded. In other words, rotating/reflecting the
board does not change which of X's or O's marks is oldest -- it only
changes *where* that oldest mark sits -- so the age-rank semantics of
each plane are preserved exactly, and the transform cannot corrupt
move-age information.

The 9-dimensional policy vector is transformed the same way: reshape
to (3, 3), apply the identical spatial transform, flatten back to 9.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

# Each entry is (name, function) where function maps a (3,3)-last-two-axes
# array to its transformed version. All operate on the *last two* axes so
# they work uniformly on a (C, 3, 3) state tensor and on a (3, 3) reshaped
# policy vector.


def _rot90(a: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(a, k=k, axes=(-2, -1))


def _flip_lr(a: np.ndarray) -> np.ndarray:
    return np.flip(a, axis=-1)


_TRANSFORMS: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("identity", lambda a: a),
    ("rot90", lambda a: _rot90(a, 1)),
    ("rot180", lambda a: _rot90(a, 2)),
    ("rot270", lambda a: _rot90(a, 3)),
    ("flip", lambda a: _flip_lr(a)),
    ("flip_rot90", lambda a: _flip_lr(_rot90(a, 1))),
    ("flip_rot180", lambda a: _flip_lr(_rot90(a, 2))),
    ("flip_rot270", lambda a: _flip_lr(_rot90(a, 3))),
]

NUM_SYMMETRIES = len(_TRANSFORMS)


def all_symmetries(
    state_tensor: np.ndarray, policy: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return all 8 (state, policy) pairs related by board symmetry.

    Args:
        state_tensor: array of shape (C, 3, 3), e.g. from
            ``FadingTicTacToe.to_tensor()``.
        policy: array of shape (9,), a probability distribution (or
            visit-count vector) over the 9 board squares in row-major
            order.

    Returns:
        A list of 8 ``(state, policy)`` pairs (including the original,
        untransformed one), each a fresh ``np.ndarray`` with the same
        shapes as the inputs.
    """
    policy_grid = np.asarray(policy, dtype=np.float32).reshape(3, 3)
    out = []
    for _name, fn in _TRANSFORMS:
        t_state = np.ascontiguousarray(fn(state_tensor))
        t_policy = np.ascontiguousarray(fn(policy_grid)).reshape(9)
        out.append((t_state, t_policy))
    return out


def random_symmetry(
    state_tensor: np.ndarray, policy: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a single randomly chosen symmetry (useful for online augmentation)."""
    idx = int(rng.integers(0, NUM_SYMMETRIES))
    _name, fn = _TRANSFORMS[idx]
    policy_grid = np.asarray(policy, dtype=np.float32).reshape(3, 3)
    t_state = np.ascontiguousarray(fn(state_tensor))
    t_policy = np.ascontiguousarray(fn(policy_grid)).reshape(9)
    return t_state, t_policy
