"""Neighbor seeds for search (section 6.6) — geometric only in Phase 2.

Local relaxation (section 7) is deferred to Phase 3; these helpers return the
raw neighbor constructions without polish.
"""

from __future__ import annotations

import numpy as np

from squarepack.constructions.removal import remove_to_n


def seed_from_larger(squares_m: np.ndarray, n: int) -> np.ndarray | None:
    """packing(n+k) minus k squares."""
    return remove_to_n(squares_m, n)


def seed_insert_one(squares_n_minus_1: np.ndarray, L: float) -> np.ndarray | None:
    """Add one axis-aligned square into the largest empty grid cell of [0, L]².

    Scans a coarse cell grid for the empty cell farthest from existing centers.
    """
    squares = np.asarray(squares_n_minus_1, dtype=np.float64)
    if L < 1.0:
        return None
    # Candidate insertion centers on a 0.5-spaced lattice inside [0.5, L-0.5].
    best = None
    best_score = -1.0
    x = 0.5
    while x <= L - 0.5 + 1e-12:
        y = 0.5
        while y <= L - 0.5 + 1e-12:
            d2 = np.min((squares[:, 0] - x) ** 2 + (squares[:, 1] - y) ** 2)
            if d2 > best_score:
                best_score = float(d2)
                best = (x, y)
            y += 0.5
        x += 0.5
    if best is None:
        return None
    extra = np.array([[best[0], best[1], 0.0]], dtype=np.float64)
    return np.vstack([squares, extra])
