"""Axis-aligned grid baseline: k = ceil(sqrt(n)), L = k."""

from __future__ import annotations

import math

import numpy as np


def grid_k(n: int) -> int:
    if n < 1:
        raise ValueError("n must be >= 1")
    return int(math.ceil(math.sqrt(n)))


def grid_packing(n: int) -> np.ndarray:
    """Return (n, 3) array of (x, y, theta) for an axis-aligned k×k grid packing.

    Places the first n cells of a row-major k×k lattice of unit squares,
    centers at (0.5 + col, 0.5 + row), angles 0. Container side is k.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    k = grid_k(n)
    squares = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        row, col = divmod(i, k)
        # Fill row-wise; for incomplete last rows this still fits in [0, k]^2.
        # Using col = i % k, row = i // k:
        squares[i, 0] = 0.5 + (i % k)
        squares[i, 1] = 0.5 + (i // k)
        squares[i, 2] = 0.0
    return squares


def grid_claimed_L(n: int) -> float:
    """Closed-form grid L (never trust for storage; always recompute via verifier)."""
    return float(grid_k(n))
