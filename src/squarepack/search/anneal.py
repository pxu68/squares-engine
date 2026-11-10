"""Simulated annealing / shrink-the-box search engine (section 7)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from squarepack.constructions.grid import grid_packing
from squarepack.geometry import bounding_box, container_length_float64, normalize_to_origin
from squarepack.search.energy import energy_and_grad, make_objective, pack_state, unpack_state
from squarepack.search.moves import LIST_ANGLES, apply_random_move, jiggle
from squarepack.verify import fast_filter


@dataclass
class AnnealResult:
    """Best feasible candidate found during a run (not yet promoted)."""

    squares: np.ndarray
    L: float
    energy: float
    seed: int
    n: int
    s_final: float
    steps: int
    elapsed_s: float
    log: list[dict[str, Any]] = field(default_factory=list)
    ok_fast: bool = False


def _clamp_into_container(squares: np.ndarray, s: float) -> np.ndarray:
    out = np.asarray(squares, dtype=np.float64).copy()
    for i in range(out.shape[0]):
        h = 0.5 * (abs(math.cos(out[i, 2])) + abs(math.sin(out[i, 2])))
        lo = h + 1e-9
        hi = s - h - 1e-9
        if lo >= hi:
            out[i, 0] = 0.5 * s
            out[i, 1] = 0.5 * s
        else:
            out[i, 0] = min(max(out[i, 0], lo), hi)
            out[i, 1] = min(max(out[i, 1], lo), hi)
    return out


def _rescale_to_container(squares: np.ndarray, s_new: float) -> np.ndarray:
    sq, L = normalize_to_origin(np.asarray(squares, dtype=np.float64))
    if L <= 1e-15:
        return _clamp_into_container(sq, s_new)
    scale = s_new / L
    sq = sq.copy()
    sq[:, 0] *= scale
    sq[:, 1] *= scale
    return _clamp_into_container(sq, s_new)


def structured_seed(n: int, rng: np.random.Generator, kind: str = "grid") -> np.ndarray:
    """Structured starts: grid / tilted rows / random."""
    if kind == "grid":
        base = grid_packing(n)
        return jiggle(base, rng, scale=0.02)
    if kind.startswith("tilt:"):
        angle = float(kind.split(":", 1)[1])
        base = grid_packing(n).copy()
        base[:, 2] = angle
        return jiggle(base, rng, scale=0.05)
    # random
    k = math.ceil(math.sqrt(n))
    s = float(k) * 1.05
    out = np.zeros((n, 3), dtype=np.float64)
    out[:, 0] = rng.uniform(0.5, s - 0.5, size=n)
    out[:, 1] = rng.uniform(0.5, s - 0.5, size=n)
    out[:, 2] = rng.choice(np.asarray(LIST_ANGLES, dtype=np.float64), size=n)
    return out


def default_init(
    n: int,
    seed_squares: np.ndarray | None,
    rng: np.random.Generator,
    best_L: float | None,
    *,
    expand: float = 1.02,
    seeded_expand: float = 1.002,
) -> tuple[np.ndarray, float]:
    """Return (squares, s0).

    Spec default: ``s0 = max(sqrt(n), 1.02 * best_known_L)``.
    For an already-feasible seed packing, keep geometry and only open a tiny
    slack (``seeded_expand``) so low-temperature seeded runs can reproduce
    records without immediately destroying them.
    """
    lower = math.sqrt(n)
    if seed_squares is not None:
        sq = np.asarray(seed_squares, dtype=np.float64).copy()
        if sq.shape[0] != n:
            raise ValueError(f"seed has {sq.shape[0]} squares, expected {n}")
        sq, L = normalize_to_origin(sq)
        known = float(best_L) if best_L is not None else float(L)
        # Keep seed geometry; container is slightly above the seed's own L.
        s0 = max(lower, float(L) * seeded_expand, known)
        # Do NOT rescale centers up to 1.02*L — that ruins record seeds.
        return _clamp_into_container(sq, s0), s0
    kind = str(
        rng.choice(
            [
                "grid",
                "tilt:0.4636476090008061",
                "tilt:0.6435011087932843",
                "tilt:0.7853981633974483",
                "random",
            ]
        )
    )
    sq = structured_seed(n, rng, kind=kind)
    sq, L = normalize_to_origin(sq)
    known = float(best_L) if best_L is not None else float(L)
    s0 = max(lower, expand * known)
    return _rescale_to_container(sq, s0), s0


def _lbfgs_minimize(squares: np.ndarray, s: float, *, maxiter: int = 80) -> tuple[np.ndarray, float]:
    n = squares.shape[0]
    obj = make_objective(n, s)
    x0 = pack_state(_clamp_into_container(squares, s))

    def fun(x: np.ndarray) -> float:
        e, _ = obj(x)
        return e

    def jac(x: np.ndarray) -> np.ndarray:
        _, g = obj(x)
        return g

    res = minimize(
        fun,
        x0,
        method="L-BFGS-B",
        jac=jac,
        options={"maxiter": maxiter, "ftol": 1e-15},
    )
    out = unpack_state(res.x, n)
    out = _clamp_into_container(out, s)
    e, _ = energy_and_grad(out, s)
    return out, float(e)


def anneal(
    n: int,
    seed: int,
    *,
    init_squares: np.ndarray | None = None,
    best_known_L: float | None = None,
    time_budget_s: float = 60.0,
    eta0: float = 1e-2,
    energy_feasible: float = 1e-24,
    temperature: float = 1e-3,
    cell_size: float = 1.5,
    max_lbfgs_iter: int = 80,
    early_stop_L: float | None = None,
    early_stop_tol: float = 1e-6,
    move_probs: Sequence[float] | None = None,
    cool_factor: float = 0.995,
    seeded_expand: float = 1.002,
    log_fn: Callable[[dict[str, Any]], None] | None = None,
) -> AnnealResult:
    """Run shrink-the-box annealing with Metropolis basin hops.

    Returns the best fast-filter-feasible candidate (lowest L) seen.
    Promotion is the caller's responsibility via store.promote.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    squares, s = default_init(
        n,
        init_squares,
        rng,
        best_known_L,
        seeded_expand=float(seeded_expand),
    )
    eta = float(eta0)
    temp = float(temperature)
    # Prefer the raw seed geometry as the incumbent when it already verifies.
    if init_squares is not None:
        seed_sq, seed_L = normalize_to_origin(np.asarray(init_squares, dtype=np.float64))
        best_sq = seed_sq
        best_L = float(seed_L)
    else:
        best_sq = squares.copy()
        best_L = container_length_float64(best_sq)
    best_e = math.inf
    steps = 0
    log: list[dict[str, Any]] = []

    def _emit(event: dict[str, Any]) -> None:
        log.append(event)
        if log_fn is not None:
            log_fn(event)

    # Record seed as candidate if already feasible.
    ff_seed = fast_filter(best_sq, cell_size=cell_size)
    if ff_seed.ok:
        best_e = 0.0
        _emit({"event": "seed_candidate", "L": best_L, "s": s, "seed": seed})

    def _hit_early() -> bool:
        return (
            early_stop_L is not None
            and best_L <= float(early_stop_L) + early_stop_tol
            and fast_filter(best_sq, cell_size=cell_size).ok
        )

    # Initial repair at current container
    squares, e = _lbfgs_minimize(squares, s, maxiter=max_lbfgs_iter)
    steps += 1
    _emit({"event": "init_minimize", "E": e, "s": s, "seed": seed})
    if e < energy_feasible:
        L_geom = container_length_float64(squares)
        ff = fast_filter(squares, L=s, cell_size=cell_size)
        if ff.ok and L_geom < best_L - 1e-15:
            best_L = L_geom
            best_sq = squares.copy()
            best_e = e
            _emit({"event": "candidate", "L": best_L, "E": e, "s": s})
    if _hit_early():
        best_sq, best_L = normalize_to_origin(best_sq)
        elapsed = time.perf_counter() - t0
        return AnnealResult(
            squares=best_sq,
            L=float(best_L),
            energy=float(best_e if math.isfinite(best_e) else e),
            seed=seed,
            n=n,
            s_final=float(s),
            steps=steps,
            elapsed_s=elapsed,
            log=log,
            ok_fast=True,
        )

    while time.perf_counter() - t0 < time_budget_s:
        if e < energy_feasible:
            # Candidate at current container; try shrink
            ff = fast_filter(squares, L=s, cell_size=cell_size)
            L_geom = container_length_float64(squares)
            if ff.ok and L_geom < best_L:
                best_L = L_geom
                best_sq = squares.copy()
                best_e = e
                _emit({"event": "candidate", "L": best_L, "E": e, "s": s})
            s_try = s * (1.0 - eta)
            if s_try < math.sqrt(n):
                eta *= 0.5
                if eta < 1e-9:
                    break
                continue
            trial = _rescale_to_container(squares, s_try)
            trial, e_try = _lbfgs_minimize(trial, s_try, maxiter=max_lbfgs_iter)
            steps += 1
            if e_try < energy_feasible:
                squares, s, e = trial, s_try, e_try
                _emit({"event": "shrink_ok", "s": s, "E": e, "eta": eta})
            else:
                eta *= 0.5
                _emit({"event": "shrink_fail", "s_try": s_try, "E": e_try, "eta": eta})
                if eta < 1e-9:
                    # Perturb and continue at current s
                    squares = apply_random_move(squares, rng, s, probs=move_probs)[0]
                    squares, e = _lbfgs_minimize(squares, s, maxiter=max_lbfgs_iter)
                    steps += 1
                    eta = float(eta0) * 0.25
            continue

        # Not feasible: perturb + minimize (Metropolis on energy)
        proposal, move = apply_random_move(squares, rng, s, probs=move_probs)
        proposal, e2 = _lbfgs_minimize(proposal, s, maxiter=max_lbfgs_iter)
        steps += 1
        dE = e2 - e
        accept = dE <= 0.0 or (temp > 0 and rng.random() < math.exp(-dE / max(temp, 1e-30)))
        if accept:
            squares, e = proposal, e2
            _emit({"event": "accept", "move": move, "E": e, "s": s})
        else:
            _emit({"event": "reject", "move": move, "E": e2, "s": s})

        # Cool slightly
        temp *= float(cool_factor)

        # Opportunistic candidate if somehow feasible
        if e < energy_feasible:
            L_geom = container_length_float64(squares)
            ff = fast_filter(squares, L=s, cell_size=cell_size)
            if ff.ok and L_geom < best_L:
                best_L = L_geom
                best_sq = squares.copy()
                best_e = e

    # Final normalize of best
    best_sq, best_L = normalize_to_origin(best_sq)
    ff = fast_filter(best_sq, cell_size=cell_size)
    elapsed = time.perf_counter() - t0
    return AnnealResult(
        squares=best_sq,
        L=float(best_L),
        energy=float(best_e if math.isfinite(best_e) else e),
        seed=seed,
        n=n,
        s_final=float(s),
        steps=steps,
        elapsed_s=elapsed,
        log=log,
        ok_fast=bool(ff.ok),
    )


def target_L_from_snapshot(n: int, snapshot_rows: dict[int, float] | None, store_L: float | None) -> float | None:
    """Prefer verified snapshot L, else store L."""
    if snapshot_rows and n in snapshot_rows:
        return float(snapshot_rows[n])
    return store_L
