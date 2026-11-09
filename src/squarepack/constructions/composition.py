"""Corner-rectangle compositions with optional small block tilt."""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np


def axis_aligned_rect(w: int, h: int, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """w by h block of axis-aligned unit squares with corner at (x0, y0)."""
    squares = np.zeros((w * h, 3), dtype=np.float64)
    i = 0
    for row in range(h):
        for col in range(w):
            squares[i] = [x0 + 0.5 + col, y0 + 0.5 + row, 0.0]
            i += 1
    return squares


def tilted_rect(
    w: int,
    h: int,
    angle: float,
    *,
    anchor_x: float,
    anchor_y: float,
) -> np.ndarray:
    """w by h block rotated by `angle` about its local origin, then translated."""
    local = axis_aligned_rect(w, h, 0.0, 0.0)
    # Shift so local corner is at origin before rotate: centers currently at 0.5+c.
    c = math.cos(angle)
    s = math.sin(angle)
    out = local.copy()
    for i in range(out.shape[0]):
        x = local[i, 0]
        y = local[i, 1]
        xr = c * x - s * y
        yr = s * x + c * y
        out[i, 0] = xr + anchor_x
        out[i, 1] = yr + anchor_y
        out[i, 2] = angle
    return out


def corner_composition(
    w1: int,
    h1: int,
    w2: int,
    h2: int,
    *,
    gap: float = 0.0,
    tilt: float = 0.0,
) -> np.ndarray | None:
    """Two rectangles in opposite corners; leftover filled with aligned squares.

    Bottom-left: w1 x h1 (optionally tilted by `tilt`).
    Top-right: w2 x h2 (tilt negated).
    Container side S = max(w1 + w2 + gap, h1 + h2 + gap) roughly; fill remaining
    cells of an S x S grid that do not overlap the two blocks' axis-aligned bboxes.
    """
    if min(w1, h1, w2, h2) < 1:
        return None
    # Untilted construction first for exact grid fill.
    if abs(tilt) < 1e-15:
        S = max(w1 + w2, h1 + h2)
        # Ensure the two rectangles fit in opposite corners without overlap.
        if w1 + w2 > S or h1 + h2 > S:
            S = max(w1 + w2, h1 + h2)
        bl = axis_aligned_rect(w1, h1, 0.0, 0.0)
        tr = axis_aligned_rect(w2, h2, float(S - w2), float(S - h2))
        occupied = set()
        for col in range(w1):
            for row in range(h1):
                occupied.add((col, row))
        for col in range(S - w2, S):
            for row in range(S - h2, S):
                occupied.add((col, row))
        fill = []
        for row in range(S):
            for col in range(S):
                if (col, row) not in occupied:
                    fill.append([0.5 + col, 0.5 + row, 0.0])
        parts = [bl, tr]
        if fill:
            parts.append(np.asarray(fill, dtype=np.float64))
        return np.vstack(parts)

    # Tilted: place two blocks and skip grid fill (fill would overlap); return
    # just the two blocks — useful as a seed, often needs search to pack gaps.
    S = max(w1 + w2, h1 + h2) + abs(gap)
    bl = tilted_rect(w1, h1, tilt, anchor_x=0.0, anchor_y=0.0)
    tr = tilted_rect(
        w2,
        h2,
        -tilt,
        anchor_x=float(S - w2 * math.cos(tilt)),
        anchor_y=float(S - h2 * math.cos(tilt)),
    )
    return np.vstack([bl, tr])


def iter_corner_params(
    *,
    max_side: int = 10,
) -> Iterator[tuple[int, int, int, int]]:
    for w1 in range(1, max_side + 1):
        for h1 in range(1, max_side + 1):
            for w2 in range(1, max_side + 1):
                for h2 in range(1, max_side + 1):
                    yield w1, h1, w2, h2


def composition_candidates_for_n(
    n: int,
    *,
    max_side: int = 8,
    limit: int = 30,
) -> list[np.ndarray]:
    from squarepack.constructions.removal import remove_to_n

    out: list[np.ndarray] = []
    for w1, h1, w2, h2 in iter_corner_params(max_side=max_side):
        pack = corner_composition(w1, h1, w2, h2)
        if pack is None or pack.shape[0] < n:
            continue
        # Prefer packs whose count is close to n to avoid huge removals.
        if pack.shape[0] > n + 40:
            continue
        trimmed = remove_to_n(pack, n)
        if trimmed is not None:
            out.append(trimmed)
            if len(out) >= limit:
                break
    return out
