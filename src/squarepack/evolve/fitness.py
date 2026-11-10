"""Fitness from verifier-approved output only (section 9)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from squarepack.constructions.grid import grid_claimed_L
from squarepack.evolve.program import SearchProgram
from squarepack.search.anneal import anneal
from squarepack.store import BestStore
from squarepack.verify import verify_full


@dataclass
class TrialResult:
    n: int
    L_found: float | None
    L_best_known: float
    grid_L: float
    verified: bool
    relative_gap: float | None
    scored: float  # contribution to mean fitness (lower is better)
    reason: str = ""


@dataclass
class FitnessReport:
    program_name: str
    ns: list[int]
    mean_score: float
    trials: list[TrialResult] = field(default_factory=list)
    n_verified: int = 0
    n_beat_grid: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "program_name": self.program_name,
            "ns": self.ns,
            "mean_score": self.mean_score,
            "n_verified": self.n_verified,
            "n_beat_grid": self.n_beat_grid,
            "trials": [t.__dict__ for t in self.trials],
        }


def score_trial(
    *,
    L_found: float | None,
    L_best_known: float,
    grid_L: float,
    verified: bool,
) -> tuple[float, str]:
    """Lower is better. Unverified / trivial-grid results get no credit."""
    if not verified or L_found is None or not math.isfinite(L_found):
        return 1.0, "unverified"
    # Trivial grid: no credit unless it beats the grid baseline.
    if L_found >= grid_L - 1e-12:
        return 1.0, "trivial_grid_no_credit"
    # Relative gap vs best known (can be negative if we beat the table).
    gap = (L_found - L_best_known) / max(L_best_known, 1e-12)
    # Clamp absurd failures
    return float(min(max(gap, -1.0), 1.0)), "ok"


def evaluate_program(
    program: SearchProgram,
    ns: Sequence[int],
    store: BestStore,
    *,
    time_budget_s: float = 8.0,
    seed0: int = 1,
    baseline: dict[int, float] | None = None,
) -> FitnessReport:
    """Run time-limited seeded anneals; score only verifier-approved outputs."""
    trials: list[TrialResult] = []
    scores: list[float] = []
    n_verified = 0
    n_beat_grid = 0

    for i, n in enumerate(ns):
        rec = store.load(int(n))
        init = None if rec is None else np.asarray(rec["squares"], dtype=np.float64)
        if baseline and int(n) in baseline:
            L_best = float(baseline[int(n)])
        elif rec is not None:
            L_best = float(rec["L"])
        else:
            L_best = float(grid_claimed_L(int(n)))
        grid_L = float(grid_claimed_L(int(n)))

        result = anneal(
            int(n),
            seed=seed0 + 17 * int(n) + i,
            init_squares=init,
            best_known_L=L_best,
            time_budget_s=time_budget_s,
            eta0=program.eta0,
            temperature=program.temperature,
            cool_factor=program.cool_factor,
            seeded_expand=program.seeded_expand,
            max_lbfgs_iter=program.max_lbfgs_iter,
            energy_feasible=program.energy_feasible,
            move_probs=program.normalized_move_probs(),
        )
        verified = False
        L_found = None
        reason = "fast_filter_fail"
        if result.ok_fast:
            report, _ = verify_full(result.squares, **store.verify_kwargs)
            verified = bool(report.ok)
            if verified and report.L is not None:
                L_found = float(report.L)
                reason = "verified"
                n_verified += 1
                if L_found < grid_L - 1e-12:
                    n_beat_grid += 1
            else:
                reason = report.reason or "verify_fail"

        scored, score_reason = score_trial(
            L_found=L_found,
            L_best_known=L_best,
            grid_L=grid_L,
            verified=verified,
        )
        trials.append(
            TrialResult(
                n=int(n),
                L_found=L_found,
                L_best_known=L_best,
                grid_L=grid_L,
                verified=verified,
                relative_gap=None
                if L_found is None
                else (L_found - L_best) / max(L_best, 1e-12),
                scored=scored,
                reason=f"{reason}|{score_reason}",
            )
        )
        scores.append(scored)

    mean = float(sum(scores) / max(len(scores), 1))
    return FitnessReport(
        program_name=program.name,
        ns=[int(n) for n in ns],
        mean_score=mean,
        trials=trials,
        n_verified=n_verified,
        n_beat_grid=n_beat_grid,
    )
