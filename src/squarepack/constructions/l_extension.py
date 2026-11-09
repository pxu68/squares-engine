"""L-extension: add an L-shaped unit-square border along two sides."""

from __future__ import annotations

import math

import numpy as np


def l_extension(squares: np.ndarray, s: float) -> tuple[np.ndarray, int] | None:
    """Extend a packing of m squares in side s by an L-border of width 1.

    Places axis-aligned unit squares along x in [s, s+1] for y-bands covering
    [0, s+1] and along y in [s, s+1] for x-bands covering [0, s], yielding
    ``m + 2*floor(s) + 1`` squares in a container of side ``s + 1``.

    Returns (new_squares, new_n) or None if s < 1.
    """
    squares = np.asarray(squares, dtype=np.float64)
    if s < 1.0 - 1e-15:
        return None
    k = int(math.floor(s + 1e-12))
    # Number of new squares: k along the bottom of the top strip? Spec:
    # m + 2*floor(s) + 1.
    # Classic construction: place floor(s) squares along the right of the old
    # square (covering height floor(s)), floor(s) along the top (covering width
    # floor(s)), and one in the corner (s, s) — total 2*floor(s)+1.
    # When s is not integer, the leftover fractional strip is left empty; the
    # new container is still s+1.
    new_rects: list[list[float]] = []
    # Right column: centers at (s + 0.5, 0.5 + i) for i = 0..k-1
    for i in range(k):
        new_rects.append([s + 0.5, 0.5 + i, 0.0])
    # Top row: centers at (0.5 + j, s + 0.5) for j = 0..k-1
    for j in range(k):
        new_rects.append([0.5 + j, s + 0.5, 0.0])
    # Corner
    new_rects.append([s + 0.5, s + 0.5, 0.0])

    extra = np.asarray(new_rects, dtype=np.float64)
    out = np.vstack([squares, extra])
    expected = squares.shape[0] + 2 * k + 1
    if out.shape[0] != expected:
        # Defensive: should not happen
        return None
    return out, int(out.shape[0])


def l_extension_new_n(m: int, s: float) -> int:
    return m + 2 * int(math.floor(s + 1e-12)) + 1
