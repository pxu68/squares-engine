"""Newton contact polish (section 10.1).

After a candidate is found: build the contact graph (near pairs + wall contacts),
resolve the active-contact system with high-precision Newton (`mpmath.findroot`),
re-verify, and return a packing only when verified L improves.

PSLQ / minimal-polynomial recovery is intentionally out of scope (section 12).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mpmath as mp
import numpy as np

from squarepack.geometry import (
    containment_margins,
    half_extent,
    normalize_to_origin,
    sep_pair,
)
from squarepack.verify import VerifyReport, verify_full


PAIR_CONTACT = 1e-7
WALL_CONTACT = 1e-7


@dataclass
class PolishResult:
    ok: bool
    squares: np.ndarray
    L: float | None
    improved: bool
    report: VerifyReport | None
    n_pair_contacts: int
    n_wall_contacts: int
    reason: str = ""


def _pair_contacts(squares: np.ndarray, tol: float = PAIR_CONTACT) -> list[tuple[int, int]]:
    n = squares.shape[0]
    out: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = sep_pair(
                float(squares[i, 0]),
                float(squares[i, 1]),
                float(squares[i, 2]),
                float(squares[j, 0]),
                float(squares[j, 1]),
                float(squares[j, 2]),
            )
            if s < tol:
                out.append((i, j))
    return out


def _wall_contacts(
    squares: np.ndarray, L: float, tol: float = WALL_CONTACT
) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for i in range(squares.shape[0]):
        left, right, bottom, top = containment_margins(
            float(squares[i, 0]), float(squares[i, 1]), float(squares[i, 2]), L
        )
        if left < tol:
            out.append((i, "left"))
        if right < tol:
            out.append((i, "right"))
        if bottom < tol:
            out.append((i, "bottom"))
        if top < tol:
            out.append((i, "top"))
    return out


def _sep_mp(sq: np.ndarray, i: int, j: int, dps: int) -> mp.mpf:
    with mp.workdps(dps):
        x1, y1, t1 = (mp.mpf(repr(float(sq[i, k]))) for k in range(3))
        x2, y2, t2 = (mp.mpf(repr(float(sq[j, k]))) for k in range(3))
        dx, dy = x2 - x1, y2 - y1
        best = mp.mpf("-inf")
        for phi in (t1, t1 + mp.pi / 2, t2, t2 + mp.pi / 2):
            ax, ay = mp.cos(phi), mp.sin(phi)
            r1 = mp.mpf("0.5") * (abs(mp.cos(phi - t1)) + abs(mp.sin(phi - t1)))
            r2 = mp.mpf("0.5") * (abs(mp.cos(phi - t2)) + abs(mp.sin(phi - t2)))
            sep = abs(ax * dx + ay * dy) - (r1 + r2)
            if sep > best:
                best = sep
        return best


def _margin_mp(sq: np.ndarray, i: int, L: mp.mpf, wall: str, dps: int) -> mp.mpf:
    with mp.workdps(dps):
        x = mp.mpf(repr(float(sq[i, 0])))
        y = mp.mpf(repr(float(sq[i, 1])))
        t = mp.mpf(repr(float(sq[i, 2])))
        h = mp.mpf("0.5") * (abs(mp.cos(t)) + abs(mp.sin(t)))
        if wall == "left":
            return x - h
        if wall == "right":
            return L - h - x
        if wall == "bottom":
            return y - h
        return L - h - y


def _newton_resolve_contacts(
    squares: np.ndarray,
    L: float,
    pair_c: list[tuple[int, int]],
    wall_c: list[tuple[int, str]],
    *,
    dps: int = 50,
    max_vars: int = 40,
) -> np.ndarray | None:
    """Adjust a small set of centers with Newton so active contacts sit at 0.

    Angles are held fixed. One square is pinned to remove rigid translation.
    If the contact system is empty or too large, returns None.
    """
    n = squares.shape[0]
    if not pair_c and not wall_c:
        return None

    involved: set[int] = set()
    for i, j in pair_c:
        involved.add(i)
        involved.add(j)
    for i, _ in wall_c:
        involved.add(i)
    idxs = sorted(involved)
    if len(idxs) < 1:
        return None
    # Cap problem size for mpmath Newton on this VM.
    if len(idxs) > max_vars:
        idxs = idxs[:max_vars]
        allowed = set(idxs)
        pair_c = [(i, j) for i, j in pair_c if i in allowed and j in allowed]
        wall_c = [(i, w) for i, w in wall_c if i in allowed]
        if not pair_c and not wall_c:
            return None

    pin = idxs[0]
    free = [i for i in idxs if i != pin]
    # Variables: (dx, dy) for each free square.
    nvar = 2 * len(free)
    residuals_desc: list[tuple[str, Any]] = []
    for i, j in pair_c:
        residuals_desc.append(("pair", (i, j)))
    for i, w in wall_c:
        residuals_desc.append(("wall", (i, w)))
    # Pad / trim to square system: prefer contact eqs; add stay-near soft pins if needed.
    while len(residuals_desc) < nvar:
        # Soft stay-near for free squares not yet constrained enough.
        k = (len(residuals_desc) // 2) % max(len(free), 1)
        axis = "x" if len(residuals_desc) % 2 == 0 else "y"
        residuals_desc.append(("stay", (free[k], axis)))
    residuals_desc = residuals_desc[:nvar]
    if nvar == 0 or not residuals_desc:
        return None

    index_of = {i: k for k, i in enumerate(free)}
    x0 = [mp.mpf(0) for _ in range(nvar)]

    def pack_sq(vec: list[mp.mpf]) -> np.ndarray:
        out = squares.copy()
        for i, k in index_of.items():
            out[i, 0] = float(mp.nstr(mp.mpf(squares[i, 0]) + vec[2 * k], n=dps))
            out[i, 1] = float(mp.nstr(mp.mpf(squares[i, 1]) + vec[2 * k + 1], n=dps))
        return out

    def residual(vec: list[mp.mpf]) -> list[mp.mpf]:
        with mp.workdps(dps):
            sq = pack_sq(vec)
            LL = mp.mpf(repr(float(L)))
            outs: list[mp.mpf] = []
            for kind, payload in residuals_desc:
                if kind == "pair":
                    i, j = payload
                    outs.append(_sep_mp(sq, i, j, dps))
                elif kind == "wall":
                    i, w = payload
                    outs.append(_margin_mp(sq, i, LL, w, dps))
                else:
                    i, axis = payload
                    k = index_of[i]
                    outs.append(vec[2 * k] if axis == "x" else vec[2 * k + 1])
            return outs

    try:
        with mp.workdps(dps):
            sol = mp.findroot(residual, x0, tol=mp.mpf("1e-20"), maxsteps=25)
            if hasattr(sol, "__len__"):
                vec = [mp.mpf(v) for v in sol]
            else:
                vec = [mp.mpf(sol)]
            return pack_sq(vec)
    except Exception:
        return None


def _try_shrink_newton(
    squares: np.ndarray,
    *,
    eta: float,
    dps: int,
) -> np.ndarray | None:
    """Shrink centers toward origin by (1-eta), Newton-resolve contacts, re-normalize."""
    sq, L = normalize_to_origin(np.asarray(squares, dtype=np.float64))
    if L <= 0:
        return None
    scale = 1.0 - eta
    if scale <= 0:
        return None
    shrunk = sq.copy()
    shrunk[:, 0] *= scale
    shrunk[:, 1] *= scale
    # New tentative container slightly above scaled L.
    L2 = float(L) * scale
    # Expand container a hair so walls aren't instantly violated before Newton.
    L_work = L2 * (1.0 + 1e-9)
    # Clamp insides roughly
    for i in range(shrunk.shape[0]):
        h = half_extent(float(shrunk[i, 2]))
        shrunk[i, 0] = min(max(float(shrunk[i, 0]), h), L_work - h)
        shrunk[i, 1] = min(max(float(shrunk[i, 1]), h), L_work - h)

    pairs = _pair_contacts(shrunk, tol=1e-5)
    walls = _wall_contacts(shrunk, L_work, tol=1e-5)
    resolved = _newton_resolve_contacts(shrunk, L_work, pairs, walls, dps=dps)
    if resolved is None:
        resolved = shrunk
    resolved, _ = normalize_to_origin(resolved)
    return resolved


def polish(
    squares: np.ndarray,
    *,
    etas: tuple[float, ...] = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9),
    dps: int = 50,
    verify_kwargs: dict[str, Any] | None = None,
) -> PolishResult:
    """Attempt Newton contact polish; return improved packing if verification wins."""
    squares = np.asarray(squares, dtype=np.float64)
    if squares.ndim != 2 or squares.shape[1] != 3:
        raise ValueError("squares must have shape (n, 3)")

    base, base_L = normalize_to_origin(squares)
    pairs0 = _pair_contacts(base)
    walls0 = _wall_contacts(base, base_L)
    vkw = dict(verify_kwargs or {})

    base_report, base_verified = verify_full(base, **vkw)
    if not base_report.ok or base_report.L is None:
        return PolishResult(
            ok=False,
            squares=base,
            L=None,
            improved=False,
            report=base_report,
            n_pair_contacts=len(pairs0),
            n_wall_contacts=len(walls0),
            reason="input failed verification",
        )
    best_sq = base_verified
    best_L = float(base_report.L)
    best_report = base_report

    improved = False
    for eta in etas:
        cand = _try_shrink_newton(best_sq, eta=eta, dps=dps)
        if cand is None:
            continue
        report, verified = verify_full(cand, **vkw)
        if report.ok and report.L is not None and float(report.L) < best_L - 0.0:
            # Strict improvement on verified L
            if float(report.L) < best_L:
                best_L = float(report.L)
                best_sq = verified
                best_report = report
                improved = True

    return PolishResult(
        ok=True,
        squares=best_sq,
        L=best_L,
        improved=improved,
        report=best_report,
        n_pair_contacts=len(pairs0),
        n_wall_contacts=len(walls0),
        reason="improved" if improved else "no improvement",
    )


def polish_and_maybe_promote(
    store: Any,
    n: int,
    squares: np.ndarray,
    *,
    seed: int | None = None,
    method: str = "newton_polish",
) -> tuple[bool, PolishResult]:
    """Polish then promote via the store if verified L improves."""
    from squarepack.store import Candidate

    result = polish(squares, verify_kwargs=getattr(store, "verify_kwargs", None) or {})
    if not result.ok or result.L is None:
        return False, result
    ok, _, _ = store.promote(
        Candidate(
            n=n,
            squares=result.squares,
            origin="searched",
            method=method,
            seed=seed,
        )
    )
    return bool(ok), result
