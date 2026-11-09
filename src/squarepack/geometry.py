"""Unit-square geometry: corners, support radius, SAT separation, bounding box."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

# Local corner offsets of a unit square centered at the origin, axis-aligned.
_CORNER_OFFSETS = np.array(
    [
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
    ],
    dtype=np.float64,
)


def rotation_matrix(theta: float) -> np.ndarray:
    """2x2 rotation matrix for angle theta (radians)."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def corners(cx: float, cy: float, theta: float) -> np.ndarray:
    """Return (4, 2) float64 array of corner positions for one unit square."""
    c = math.cos(theta)
    s = math.sin(theta)
    # R @ offset for each local corner
    xs = cx + _CORNER_OFFSETS[:, 0] * c - _CORNER_OFFSETS[:, 1] * s
    ys = cy + _CORNER_OFFSETS[:, 0] * s + _CORNER_OFFSETS[:, 1] * c
    return np.column_stack((xs, ys))


def corners_batch(squares: np.ndarray) -> np.ndarray:
    """Corners for an (n, 3) array of (x, y, theta). Shape (n, 4, 2)."""
    squares = np.asarray(squares, dtype=np.float64)
    n = squares.shape[0]
    out = np.empty((n, 4, 2), dtype=np.float64)
    for i in range(n):
        out[i] = corners(float(squares[i, 0]), float(squares[i, 1]), float(squares[i, 2]))
    return out


def support_radius(phi: float, t: float) -> float:
    """Support radius of a unit square at angle t along axis angle phi.

    r(phi, t) = 0.5 * (|cos(phi - t)| + |sin(phi - t)|)
    """
    d = phi - t
    return 0.5 * (abs(math.cos(d)) + abs(math.sin(d)))


def half_extent(theta: float) -> float:
    """Axis-aligned half-width of a unit square at angle theta (same on x and y)."""
    return 0.5 * (abs(math.cos(theta)) + abs(math.sin(theta)))


def sep_axis(dx: float, dy: float, phi: float, t1: float, t2: float) -> float:
    """Separation along unit axis at angle phi for two unit squares."""
    ax = math.cos(phi)
    ay = math.sin(phi)
    proj = abs(ax * dx + ay * dy)
    return proj - (support_radius(phi, t1) + support_radius(phi, t2))


def sep_pair(
    c1x: float,
    c1y: float,
    t1: float,
    c2x: float,
    c2y: float,
    t2: float,
) -> float:
    """SAT separation of two unit squares. >= 0 means separated or touching."""
    dx = c2x - c1x
    dy = c2y - c1y
    axes = (t1, t1 + 0.5 * math.pi, t2, t2 + 0.5 * math.pi)
    best = -math.inf
    for phi in axes:
        s = sep_axis(dx, dy, phi, t1, t2)
        if s > best:
            best = s
    return best


def sep_pair_from_arrays(a: Iterable[float], b: Iterable[float]) -> float:
    """Convenience wrapper: each argument is (x, y, theta)."""
    a = tuple(a)
    b = tuple(b)
    return sep_pair(a[0], a[1], a[2], b[0], b[1], b[2])


def bounding_box(squares: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box over all corners: (xmin, xmax, ymin, ymax)."""
    squares = np.asarray(squares, dtype=np.float64)
    if squares.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    all_corners = corners_batch(squares).reshape(-1, 2)
    xmin = float(np.min(all_corners[:, 0]))
    xmax = float(np.max(all_corners[:, 0]))
    ymin = float(np.min(all_corners[:, 1]))
    ymax = float(np.max(all_corners[:, 1]))
    return xmin, xmax, ymin, ymax


def container_length_float64(squares: np.ndarray) -> float:
    """L = max(xmax - xmin, ymax - ymin) in float64 (not the verified L)."""
    xmin, xmax, ymin, ymax = bounding_box(squares)
    return max(xmax - xmin, ymax - ymin)


def containment_margins(
    cx: float,
    cy: float,
    theta: float,
    L: float,
) -> tuple[float, float, float, float]:
    """Margins to the four walls of [0, L]^2: (left, right, bottom, top)."""
    h = half_extent(theta)
    return (cx - h, L - h - cx, cy - h, L - h - cy)


def normalize_to_origin(squares: np.ndarray) -> tuple[np.ndarray, float]:
    """Translate so bounding box starts at origin; return (squares, L_float64)."""
    squares = np.asarray(squares, dtype=np.float64).copy()
    xmin, xmax, ymin, ymax = bounding_box(squares)
    squares[:, 0] -= xmin
    squares[:, 1] -= ymin
    L = max(xmax - xmin, ymax - ymin)
    return squares, L
