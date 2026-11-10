"""Basin-hopping perturbation moves (section 7)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from squarepack.geometry import half_extent

LIST_ANGLES = (
    0.0,
    math.atan(0.5),
    math.atan(0.75),
    0.25 * math.pi,
    0.5 * math.pi,
)


def translate(squares: np.ndarray, rng: np.random.Generator, scale: float = 0.15) -> np.ndarray:
    out = np.asarray(squares, dtype=np.float64).copy()
    i = int(rng.integers(0, out.shape[0]))
    out[i, 0] += float(rng.normal(0.0, scale))
    out[i, 1] += float(rng.normal(0.0, scale))
    return out


def rotate(squares: np.ndarray, rng: np.random.Generator, scale: float = 0.2) -> np.ndarray:
    out = np.asarray(squares, dtype=np.float64).copy()
    i = int(rng.integers(0, out.shape[0]))
    out[i, 2] += float(rng.normal(0.0, scale))
    return out


def jiggle(squares: np.ndarray, rng: np.random.Generator, scale: float = 0.05) -> np.ndarray:
    out = np.asarray(squares, dtype=np.float64).copy()
    out[:, 0] += rng.normal(0.0, scale, size=out.shape[0])
    out[:, 1] += rng.normal(0.0, scale, size=out.shape[0])
    out[:, 2] += rng.normal(0.0, scale * 0.5, size=out.shape[0])
    return out


def swap_orientation(
    squares: np.ndarray,
    rng: np.random.Generator,
    angles: Sequence[float] = LIST_ANGLES,
) -> np.ndarray:
    out = np.asarray(squares, dtype=np.float64).copy()
    i = int(rng.integers(0, out.shape[0]))
    out[i, 2] = float(angles[int(rng.integers(0, len(angles)))])
    return out


def relocate(squares: np.ndarray, rng: np.random.Generator, s: float) -> np.ndarray:
    """Remove one square and reinsert into the largest free hole on a coarse lattice."""
    out = np.asarray(squares, dtype=np.float64).copy()
    n = out.shape[0]
    if n == 0 or s < 1.0:
        return out
    i = int(rng.integers(0, n))
    rest = np.delete(out, i, axis=0)
    best = None
    best_score = -1.0
    step = 0.5
    x = 0.5
    while x <= s - 0.5 + 1e-12:
        y = 0.5
        while y <= s - 0.5 + 1e-12:
            if rest.shape[0] == 0:
                d2 = 1e9
            else:
                d2 = float(np.min((rest[:, 0] - x) ** 2 + (rest[:, 1] - y) ** 2))
            if d2 > best_score:
                best_score = d2
                best = (x, y)
            y += step
        x += step
    if best is None:
        return out
    theta = float(angles_choice(rng))
    h = half_extent(theta)
    bx = min(max(best[0], h), s - h)
    by = min(max(best[1], h), s - h)
    out[i] = (bx, by, theta)
    return out


def angles_choice(rng: np.random.Generator) -> float:
    return float(LIST_ANGLES[int(rng.integers(0, len(LIST_ANGLES)))])


def shift_row(squares: np.ndarray, rng: np.random.Generator, scale: float = 0.25) -> np.ndarray:
    """Shift all squares whose centers share a coarse y-bin."""
    out = np.asarray(squares, dtype=np.float64).copy()
    if out.shape[0] == 0:
        return out
    bins = np.round(out[:, 1] * 2.0) / 2.0
    uniq = np.unique(bins)
    if uniq.size == 0:
        return out
    target = float(uniq[int(rng.integers(0, uniq.size))])
    dx = float(rng.normal(0.0, scale))
    mask = np.isclose(bins, target)
    out[mask, 0] += dx
    return out


MOVE_NAMES = (
    "translate",
    "rotate",
    "jiggle",
    "swap_orientation",
    "relocate",
    "shift_row",
)


def apply_random_move(
    squares: np.ndarray,
    rng: np.random.Generator,
    s: float,
    *,
    probs: Sequence[float] | None = None,
) -> tuple[np.ndarray, str]:
    names = MOVE_NAMES
    p = np.asarray(probs if probs is not None else (0.25, 0.2, 0.2, 0.1, 0.15, 0.1), dtype=np.float64)
    p = p / p.sum()
    name = str(rng.choice(names, p=p))
    if name == "translate":
        return translate(squares, rng), name
    if name == "rotate":
        return rotate(squares, rng), name
    if name == "jiggle":
        return jiggle(squares, rng), name
    if name == "swap_orientation":
        return swap_orientation(squares, rng), name
    if name == "relocate":
        return relocate(squares, rng, s), name
    return shift_row(squares, rng), name
