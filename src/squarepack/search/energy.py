"""Penalty energy and analytic gradients for fixed-container annealing (section 7)."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from squarepack.geometry import half_extent, sep_pair, support_radius


def _cell_near_pairs(squares: np.ndarray, cell_size: float = 1.5) -> list[tuple[int, int]]:
    n = squares.shape[0]
    if n < 2:
        return []
    xmin = float(np.min(squares[:, 0]))
    ymin = float(np.min(squares[:, 1]))
    inv = 1.0 / cell_size
    cells: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        cx = int(math.floor((squares[i, 0] - xmin) * inv))
        cy = int(math.floor((squares[i, 1] - ymin) * inv))
        cells.setdefault((cx, cy), []).append(i)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for (cx, cy), idxs in cells.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if i > j:
                    i, j = j, i
                if (i, j) not in seen:
                    seen.add((i, j))
                    pairs.append((i, j))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                other = cells.get((cx + dx, cy + dy))
                if not other:
                    continue
                for i in idxs:
                    for j in other:
                        a, b = (i, j) if i < j else (j, i)
                        if (a, b) not in seen:
                            seen.add((a, b))
                            pairs.append((a, b))
    return pairs


def _active_sep_axis(
    dx: float, dy: float, t1: float, t2: float
) -> tuple[float, float, float]:
    """Return (sep, ax, ay) for the maximizing SAT axis."""
    best = -math.inf
    best_ax = 1.0
    best_ay = 0.0
    for phi in (t1, t1 + 0.5 * math.pi, t2, t2 + 0.5 * math.pi):
        ax = math.cos(phi)
        ay = math.sin(phi)
        proj = abs(ax * dx + ay * dy)
        s = proj - (support_radius(phi, t1) + support_radius(phi, t2))
        if s > best:
            best = s
            # Signed projection direction for gradient of |a·d|
            sign = 1.0 if (ax * dx + ay * dy) >= 0.0 else -1.0
            best_ax = ax * sign
            best_ay = ay * sign
    return best, best_ax, best_ay


def energy_and_grad(
    squares: np.ndarray,
    s: float,
    *,
    cell_size: float = 1.5,
) -> tuple[float, np.ndarray]:
    """Penalty energy and (n,3) gradient w.r.t. (x,y,theta) at container side s.

    E = sum max(0, -sep)^2 + sum wall_violation^2
    Gradients use the active SAT axis (subgradient when multiple axes tie).
    """
    squares = np.asarray(squares, dtype=np.float64)
    n = squares.shape[0]
    g = np.zeros((n, 3), dtype=np.float64)
    e = 0.0

    pairs = _cell_near_pairs(squares, cell_size=cell_size)
    for i, j in pairs:
        x1, y1, t1 = map(float, squares[i])
        x2, y2, t2 = map(float, squares[j])
        dx = x2 - x1
        dy = y2 - y1
        sep, ax, ay = _active_sep_axis(dx, dy, t1, t2)
        if sep >= 0.0:
            continue
        pen = -sep
        e += pen * pen
        # dE/dsep = 2*(-sep)*(-1) wait: E = pen^2, pen=-sep, dE/dsep = 2*pen*(-1) = 2*sep
        # Force on centers: d sep / d x1 = -ax, d sep / d x2 = ax
        coef = 2.0 * sep  # = -2 * pen
        # ∂E/∂x1 = coef * ∂sep/∂x1 = coef * (-ax)
        g[i, 0] += coef * (-ax)
        g[i, 1] += coef * (-ay)
        g[j, 0] += coef * ax
        g[j, 1] += coef * ay
        # Angle gradients via finite difference on support radii of active axis only —
        # use a small central difference for robustness on abs(cos/sin).
        eps = 1e-7
        for idx, t0, sign_t in ((i, t1, 1.0), (j, t2, 1.0)):
            # Perturb this square's angle only
            if idx == i:
                s_p = sep_pair(x1, y1, t1 + eps, x2, y2, t2)
                s_m = sep_pair(x1, y1, t1 - eps, x2, y2, t2)
            else:
                s_p = sep_pair(x1, y1, t1, x2, y2, t2 + eps)
                s_m = sep_pair(x1, y1, t1, x2, y2, t2 - eps)
            dsep_dt = (s_p - s_m) / (2.0 * eps)
            g[idx, 2] += coef * dsep_dt

    # Wall penalties
    for i in range(n):
        x, y, t = map(float, squares[i])
        h = half_extent(t)
        # left: h - x  (violation if > 0)
        # right: x - (s - h)
        # bottom: h - y
        # top: y - (s - h)
        for viol, d_x, d_y, d_t_fn in (
            (h - x, -1.0, 0.0, True),
            (x - (s - h), 1.0, 0.0, True),
            (h - y, 0.0, -1.0, True),
            (y - (s - h), 0.0, 1.0, True),
        ):
            if viol <= 0.0:
                continue
            e += viol * viol
            coef = 2.0 * viol
            g[i, 0] += coef * d_x
            g[i, 1] += coef * d_y
            # dh/dtheta = 0.5 * (-sin*sgn(cos) + cos*sgn(sin)) approx via FD
            eps = 1e-7
            hp = half_extent(t + eps)
            hm = half_extent(t - eps)
            dh = (hp - hm) / (2.0 * eps)
            # left/bottom: viol = h - ·  => ∂viol/∂t = dh
            # right/top: viol = · - (s - h) => ∂viol/∂t = dh
            g[i, 2] += coef * dh

    return float(e), g


def pack_state(squares: np.ndarray) -> np.ndarray:
    """Flatten (n,3) -> (3n,) for SciPy."""
    return np.asarray(squares, dtype=np.float64).ravel()


def unpack_state(vec: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(vec, dtype=np.float64).reshape(n, 3)


def make_objective(
    n: int,
    s: float,
    *,
    cell_size: float = 1.5,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    def obj(vec: np.ndarray) -> tuple[float, np.ndarray]:
        sq = unpack_state(vec, n)
        e, g = energy_and_grad(sq, s, cell_size=cell_size)
        return e, g.ravel()

    return obj
