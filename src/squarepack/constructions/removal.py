"""Monotonic removal: packing(m) minus squares yields candidates for n < m."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _boundary_score(squares: np.ndarray, L: float) -> np.ndarray:
    """Higher score = closer to the container boundary (prefer remove first)."""
    scores = np.empty(squares.shape[0], dtype=np.float64)
    for i, (x, y, _t) in enumerate(squares):
        scores[i] = min(float(x), float(y), L - float(x), L - float(y))
    # Smaller margin to wall => higher removal priority => sort ascending margin.
    return scores


def removal_order(squares: np.ndarray, L: float | None = None) -> np.ndarray:
    """Indices to remove first: boundary squares, then by distance from center."""
    squares = np.asarray(squares, dtype=np.float64)
    n = squares.shape[0]
    if L is None:
        L = float(max(np.max(squares[:, 0] + 0.5), np.max(squares[:, 1] + 0.5)))
    wall = _boundary_score(squares, float(L))
    cx = float(np.mean(squares[:, 0]))
    cy = float(np.mean(squares[:, 1]))
    dist = (squares[:, 0] - cx) ** 2 + (squares[:, 1] - cy) ** 2
    # Prefer small wall margin; break ties by larger distance from center (outer first).
    order = np.lexsort((-dist, wall))
    return order


def remove_to_n(squares: np.ndarray, n: int, L: float | None = None) -> np.ndarray | None:
    """Return a packing with exactly n squares by deleting from a larger packing."""
    squares = np.asarray(squares, dtype=np.float64)
    m = squares.shape[0]
    if n < 1 or n > m:
        return None
    if n == m:
        return squares.copy()
    order = removal_order(squares, L)
    drop = set(int(i) for i in order[: m - n])
    keep = [i for i in range(m) if i not in drop]
    return squares[keep]


def removal_candidates_from_best(
    best_squares_by_n: dict[int, np.ndarray],
    best_L_by_n: dict[int, float],
    *,
    n_values: Iterable[int],
    max_source_gap: int = 50,
) -> list[tuple[int, np.ndarray, str]]:
    """For each n, try removing from larger m packings (m <= n + max_source_gap).

    Returns list of (n, squares, method_tag) candidates (not yet verified).
    """
    out: list[tuple[int, np.ndarray, str]] = []
    sources = sorted(best_squares_by_n)
    for n in n_values:
        for m in sources:
            if m <= n or m > n + max_source_gap:
                continue
            cand = remove_to_n(best_squares_by_n[m], n, best_L_by_n.get(m))
            if cand is not None:
                out.append((n, cand, f"removal_from_{m}"))
    return out
