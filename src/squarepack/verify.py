"""Verifier: fast filter + full verification with dilation and high precision."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mpmath as mp
import numpy as np

from squarepack.geometry import (
    bounding_box,
    containment_margins,
    half_extent,
    normalize_to_origin,
    sep_pair,
)


@dataclass
class VerifyReport:
    """Structured verification report (section 4.3)."""

    ok: bool
    L: float | None
    min_sep: float | None
    min_margin: float | None
    worst_pair: tuple[int, int] | None
    worst_square: int | None
    delta: float
    float64_ok: bool
    mp50_ok: bool
    reason: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "L": self.L,
            "min_sep": self.min_sep,
            "min_margin": self.min_margin,
            "worst_pair": list(self.worst_pair) if self.worst_pair is not None else None,
            "worst_square": self.worst_square,
            "delta": self.delta,
            "float64_ok": self.float64_ok,
            "mp50_ok": self.mp50_ok,
            "reason": self.reason,
            **self.extras,
        }


def _cell_pairs(squares: np.ndarray, cell_size: float) -> list[tuple[int, int]]:
    """Near pairs via a uniform cell list. Cell size ~1.5; never O(n^2) at n=1000."""
    n = squares.shape[0]
    if n < 2:
        return []
    # Map centers into cells
    xmin = float(np.min(squares[:, 0]))
    ymin = float(np.min(squares[:, 1]))
    # Support diameter of a unit square is sqrt(2); cell_size ~1.5 covers near neighbors
    inv = 1.0 / cell_size
    cells: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        cx = int(math.floor((squares[i, 0] - xmin) * inv))
        cy = int(math.floor((squares[i, 1] - ymin) * inv))
        cells.setdefault((cx, cy), []).append(i)

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    neighbor_offsets = [
        (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
    ]
    for (cx, cy), idxs in cells.items():
        # pairs inside the same cell
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if i > j:
                    i, j = j, i
                key = (i, j)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
        # pairs with neighboring cells (lexicographic to avoid duplicates)
        for dx, dy in neighbor_offsets:
            if dx == 0 and dy == 0:
                continue
            other = cells.get((cx + dx, cy + dy))
            if not other:
                continue
            for i in idxs:
                for j in other:
                    a, b = (i, j) if i < j else (j, i)
                    key = (a, b)
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
    return pairs


def _all_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _pair_seps(
    squares: np.ndarray,
    pairs: list[tuple[int, int]],
) -> tuple[float, tuple[int, int] | None, list[tuple[int, int, float]]]:
    """Return (min_sep, worst_pair, list of (i,j,sep) for all checked pairs)."""
    if not pairs:
        return math.inf, None, []
    min_sep = math.inf
    worst: tuple[int, int] | None = None
    details: list[tuple[int, int, float]] = []
    for i, j in pairs:
        s = sep_pair(
            float(squares[i, 0]),
            float(squares[i, 1]),
            float(squares[i, 2]),
            float(squares[j, 0]),
            float(squares[j, 1]),
            float(squares[j, 2]),
        )
        details.append((i, j, s))
        if s < min_sep:
            min_sep = s
            worst = (i, j)
    return min_sep, worst, details


def _min_containment(
    squares: np.ndarray,
    L: float,
) -> tuple[float, int | None]:
    min_m = math.inf
    worst_i: int | None = None
    for i in range(squares.shape[0]):
        left, right, bottom, top = containment_margins(
            float(squares[i, 0]),
            float(squares[i, 1]),
            float(squares[i, 2]),
            L,
        )
        m = min(left, right, bottom, top)
        if m < min_m:
            min_m = m
            worst_i = i
    return min_m, worst_i


def fast_filter(
    squares: np.ndarray,
    L: float | None = None,
    *,
    sep_tol: float = 1e-9,
    contain_tol: float = 1e-9,
    cell_size: float = 1.5,
) -> VerifyReport:
    """Fast float64 filter with cell list (section 4.3). Not for promotion."""
    squares = np.asarray(squares, dtype=np.float64)
    n = squares.shape[0]
    if n == 0:
        return VerifyReport(
            ok=True,
            L=0.0,
            min_sep=math.inf,
            min_margin=math.inf,
            worst_pair=None,
            worst_square=None,
            delta=0.0,
            float64_ok=True,
            mp50_ok=True,
            reason="empty",
        )

    if L is None:
        xmin, xmax, ymin, ymax = bounding_box(squares)
        L = max(xmax - xmin, ymax - ymin)

    pairs = _cell_pairs(squares, cell_size)
    min_sep, worst_pair, _ = _pair_seps(squares, pairs)
    min_margin, worst_sq = _min_containment(squares, float(L))

    ok = (min_sep >= -sep_tol) and (min_margin >= -contain_tol)
    reason = ""
    if min_sep < -sep_tol:
        reason = f"overlap pair {worst_pair} sep={min_sep}"
    elif min_margin < -contain_tol:
        reason = f"containment square {worst_sq} margin={min_margin}"

    return VerifyReport(
        ok=ok,
        L=float(L),
        min_sep=float(min_sep) if math.isfinite(min_sep) else None,
        min_margin=float(min_margin) if math.isfinite(min_margin) else None,
        worst_pair=worst_pair,
        worst_square=worst_sq,
        delta=0.0,
        float64_ok=ok,
        mp50_ok=False,
        reason=reason,
    )


def _next_float_up(x: float) -> float:
    """Round upward to the next representable float64 (section 4.4)."""
    if not math.isfinite(x):
        return x
    # numpy nextafter toward +inf
    return float(np.nextafter(np.float64(x), np.float64(np.inf)))


def compute_L_verified(squares: np.ndarray, *, dps: int = 50) -> float:
    """High-precision L from corners, stored as float64 rounded up."""
    squares = np.asarray(squares, dtype=np.float64)
    with mp.workdps(dps):
        xmin = mp.mpf("inf")
        xmax = mp.mpf("-inf")
        ymin = mp.mpf("inf")
        ymax = mp.mpf("-inf")
        half = mp.mpf("0.5")
        for i in range(squares.shape[0]):
            cx = mp.mpf(float(squares[i, 0]))
            cy = mp.mpf(float(squares[i, 1]))
            th = mp.mpf(float(squares[i, 2]))
            c = mp.cos(th)
            s = mp.sin(th)
            for ox, oy in ((-half, -half), (half, -half), (half, half), (-half, half)):
                x = cx + ox * c - oy * s
                y = cy + ox * s + oy * c
                if x < xmin:
                    xmin = x
                if x > xmax:
                    xmax = x
                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y
        L_hp = mp.mpf(max(xmax - xmin, ymax - ymin))
        # Convert to float64 via string to avoid double rounding surprises, then round up.
        L_f = float(L_hp)
    # Ensure stored L is never below the high-precision value.
    # If float(L_hp) rounded down, nextafter bumps it up.
    L_up = _next_float_up(L_f)
    # If float already rounded up or exact, nextafter still goes one ulp higher —
    # section 4.4 says "rounded up to the next representable double", meaning
    # ceil in the float lattice: if exact, keep; if truncated down, bump.
    # Safer interpretation used here: always take nextafter so stored L is a
    # strict upper bound on the high-precision value, then verify containment
    # against that L. Documented in ASSUMPTIONS.md.
    # Re-check: if float(L_hp) >= L_hp (when compared at high prec), keep float;
    # else bump.
    with mp.workdps(dps):
        if mp.mpf(repr(L_f)) >= L_hp:
            return L_f
        return L_up


def _support_radius_mp(phi: mp.mpf, t: mp.mpf) -> mp.mpf:
    d = phi - t
    return mp.mpf("0.5") * (abs(mp.cos(d)) + abs(mp.sin(d)))


def _sep_pair_mp(
    c1x: float,
    c1y: float,
    t1: float,
    c2x: float,
    c2y: float,
    t2: float,
    *,
    dps: int = 50,
) -> float:
    """High-precision SAT separation treating float64 inputs as exact."""
    with mp.workdps(dps):
        x1 = mp.mpf(repr(float(c1x)))
        y1 = mp.mpf(repr(float(c1y)))
        a1 = mp.mpf(repr(float(t1)))
        x2 = mp.mpf(repr(float(c2x)))
        y2 = mp.mpf(repr(float(c2y)))
        a2 = mp.mpf(repr(float(t2)))
        dx = x2 - x1
        dy = y2 - y1
        axes = (a1, a1 + mp.pi / 2, a2, a2 + mp.pi / 2)
        best = mp.mpf("-inf")
        for phi in axes:
            ax = mp.cos(phi)
            ay = mp.sin(phi)
            proj = abs(ax * dx + ay * dy)
            sep = proj - (_support_radius_mp(phi, a1) + _support_radius_mp(phi, a2))
            if sep > best:
                best = sep
        return float(best)


def _containment_margins_mp(
    cx: float,
    cy: float,
    theta: float,
    L: float,
    *,
    dps: int = 50,
) -> float:
    with mp.workdps(dps):
        x = mp.mpf(repr(float(cx)))
        y = mp.mpf(repr(float(cy)))
        th = mp.mpf(repr(float(theta)))
        LL = mp.mpf(repr(float(L)))
        h = mp.mpf("0.5") * (abs(mp.cos(th)) + abs(mp.sin(th)))
        m = min(x - h, LL - h - x, y - h, LL - h - y)
        return float(m)


def dilate(squares: np.ndarray, delta: float) -> np.ndarray:
    """Scale centers by (1 + delta); angles unchanged (section 4.5)."""
    out = np.asarray(squares, dtype=np.float64).copy()
    scale = 1.0 + delta
    out[:, 0] *= scale
    out[:, 1] *= scale
    return out


def _float64_full_check(
    squares: np.ndarray,
    L: float,
    *,
    contact_threshold: float = 1e-6,
) -> tuple[bool, float, float, tuple[int, int] | None, int | None, list[tuple[int, int, float]]]:
    """Strict float64 check: sep >= 0 and margins >= 0 for all pairs."""
    n = squares.shape[0]
    pairs = _all_pairs(n)
    min_sep, worst_pair, details = _pair_seps(squares, pairs)
    min_margin, worst_sq = _min_containment(squares, L)
    ok = (min_sep >= 0.0) and (min_margin >= 0.0)
    return ok, min_sep, min_margin, worst_pair, worst_sq, details


def _mp_contact_check(
    squares: np.ndarray,
    L: float,
    details: list[tuple[int, int, float]],
    *,
    contact_threshold: float = 1e-6,
    dps: int = 50,
) -> tuple[bool, float, float, tuple[int, int] | None, int | None]:
    """Re-check near-contact pairs and all containment at 50 digits."""
    min_sep = math.inf
    worst_pair: tuple[int, int] | None = None
    for i, j, s64 in details:
        if s64 > contact_threshold:
            # safely separated in float64; keep as lower bound on min_sep
            if s64 < min_sep:
                min_sep = s64
                worst_pair = (i, j)
            continue
        s = _sep_pair_mp(
            float(squares[i, 0]),
            float(squares[i, 1]),
            float(squares[i, 2]),
            float(squares[j, 0]),
            float(squares[j, 1]),
            float(squares[j, 2]),
            dps=dps,
        )
        if s < min_sep:
            min_sep = s
            worst_pair = (i, j)
        if s < 0.0:
            return False, s, math.nan, (i, j), None

    min_margin = math.inf
    worst_sq: int | None = None
    for i in range(squares.shape[0]):
        m = _containment_margins_mp(
            float(squares[i, 0]),
            float(squares[i, 1]),
            float(squares[i, 2]),
            L,
            dps=dps,
        )
        if m < min_margin:
            min_margin = m
            worst_sq = i
        if m < 0.0:
            return False, min_sep, m, worst_pair, i

    return True, float(min_sep), float(min_margin), worst_pair, worst_sq


def verify_full(
    squares: np.ndarray,
    *,
    delta: float | None = None,
    delta_default: float = 1e-12,
    delta_max: float = 1e-6,
    delta_growth: float = 10.0,
    normalize: bool = True,
    contact_threshold: float = 1e-6,
    dps: int = 50,
    L_hint: float | None = None,
) -> tuple[VerifyReport, np.ndarray]:
    """Full verification with dilation. Returns (report, possibly-dilated squares).

    Tries the smallest delta in {0, delta_default, delta_default*growth, ...}
    that passes both float64 (no tolerance) and mpmath checks after dilation.
    """
    squares = np.asarray(squares, dtype=np.float64).copy()
    if squares.ndim != 2 or squares.shape[1] != 3:
        raise ValueError("squares must have shape (n, 3)")

    if normalize:
        squares, _ = normalize_to_origin(squares)

    # Candidate deltas: try 0 first (already strictly feasible), then grow.
    if delta is not None:
        deltas = [float(delta)]
    else:
        deltas = [0.0]
        d = float(delta_default)
        while d <= delta_max + 1e-30:
            deltas.append(d)
            d *= delta_growth

    last_reason = "no delta succeeded"
    for d in deltas:
        cand = dilate(squares, d) if d > 0.0 else squares.copy()
        if normalize and d > 0.0:
            # After dilation from origin-normalized packing, bbox still starts near 0
            # but re-normalize to keep container at [0,L]^2.
            cand, _ = normalize_to_origin(cand)

        L = compute_L_verified(cand, dps=dps)
        if L_hint is not None and L_hint > L:
            # Allow caller to propose a larger container; still recompute from geometry.
            pass

        f64_ok, min_sep, min_margin, worst_pair, worst_sq, details = _float64_full_check(
            cand, L, contact_threshold=contact_threshold
        )
        if not f64_ok:
            last_reason = (
                f"float64 fail at delta={d}: sep={min_sep}, margin={min_margin}, "
                f"pair={worst_pair}, sq={worst_sq}"
            )
            continue

        mp_ok, mp_sep, mp_margin, mp_pair, mp_sq = _mp_contact_check(
            cand,
            L,
            details,
            contact_threshold=contact_threshold,
            dps=dps,
        )
        if not mp_ok:
            last_reason = (
                f"mpmath fail at delta={d}: sep={mp_sep}, margin={mp_margin}, "
                f"pair={mp_pair}, sq={mp_sq}"
            )
            continue

        # Use the more conservative (smaller) of the two min seps/margins for the report.
        report_sep = min(min_sep, mp_sep)
        report_margin = min(min_margin, mp_margin)
        sep_out = float(report_sep) if math.isfinite(report_sep) else None
        margin_out = float(report_margin) if math.isfinite(report_margin) else None
        report = VerifyReport(
            ok=True,
            L=L,
            min_sep=sep_out,
            min_margin=margin_out,
            worst_pair=mp_pair if mp_sep <= min_sep else worst_pair,
            worst_square=mp_sq if mp_margin <= min_margin else worst_sq,
            delta=d,
            float64_ok=True,
            mp50_ok=True,
            reason="ok",
        )
        return report, cand

    fail = VerifyReport(
        ok=False,
        L=None,
        min_sep=None,
        min_margin=None,
        worst_pair=None,
        worst_square=None,
        delta=deltas[-1] if deltas else 0.0,
        float64_ok=False,
        mp50_ok=False,
        reason=last_reason,
    )
    return fail, squares


def load_config_verify_kwargs(config: dict[str, Any] | None) -> dict[str, Any]:
    """Extract verify_full kwargs from config.yaml structure."""
    if not config:
        return {}
    g = config.get("geometry", {})
    v = config.get("verification", {})
    return {
        "delta_default": float(v.get("delta_default", 1e-12)),
        "delta_max": float(v.get("delta_max", 1e-6)),
        "delta_growth": float(v.get("delta_growth", 10.0)),
        "normalize": bool(v.get("normalize", True)),
        "contact_threshold": float(g.get("contact_sep_threshold", 1e-6)),
        "dps": int(v.get("mpmath_dps", 50)),
    }
