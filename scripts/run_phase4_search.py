#!/usr/bin/env python3
"""Phase 4: sampled seeded low-temperature search + polish over large-n ranges.

Full sweeps of n=51..324 and 325..1000 exceed this VM's wall-clock budget, so
we draw a stratified sample (documented in docs/BENCH.md / ASSUMPTIONS.md),
run short seeded anneals, polish, and promote only on verified improvement.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from squarepack.polish import polish  # noqa: E402
from squarepack.scheduler import default_skip_n  # noqa: E402
from squarepack.search.anneal import anneal  # noqa: E402
from squarepack.search.basin import record_run  # noqa: E402
from squarepack.store import BestStore, Candidate, load_config  # noqa: E402


def load_baseline(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["n"])] = row
    return out


def stratified_sample(
    ns: list[int],
    k: int,
    rng: np.random.Generator,
) -> list[int]:
    """Evenly spaced sample across sorted ns, with slight jitter."""
    if k >= len(ns):
        return list(ns)
    if k <= 0:
        return []
    chosen: list[int] = []
    for i in range(k):
        lo = int(i * len(ns) / k)
        hi = int((i + 1) * len(ns) / k)
        hi = max(hi, lo + 1)
        pick = int(rng.integers(lo, hi))
        chosen.append(ns[pick])
    # unique, sorted
    return sorted(set(chosen))


def candidate_ns(n_lo: int, n_hi: int, skip_near: bool) -> list[int]:
    return [
        n
        for n in range(n_lo, n_hi + 1)
        if not default_skip_n(n, skip_near_square=skip_near)
    ]


def run_one(
    store: BestStore,
    n: int,
    *,
    seed: int,
    budget_s: float,
    logs_dir: Path,
    baseline_L: float,
    do_polish: bool,
) -> dict:
    rec = store.load(n)
    init = None if rec is None else np.asarray(rec["squares"], dtype=np.float64)
    best_L = store.current_L(n)
    t0 = time.perf_counter()
    result = anneal(
        n,
        seed,
        init_squares=init,
        best_known_L=best_L,
        time_budget_s=budget_s,
        temperature=1e-4,
        eta0=5e-3,
        early_stop_L=None,  # allow shrink attempts within budget
    )
    squares = result.squares
    method = "anneal"
    if do_polish:
        pr = polish(squares, verify_kwargs=store.verify_kwargs)
        if pr.ok and pr.L is not None:
            squares = pr.squares
            method = "anneal+newton_polish" if pr.improved else "anneal_polish_checked"

    promoted = False
    verified_L = None
    if result.ok_fast or do_polish:
        ok, report, _ = store.promote(
            Candidate(
                n=n,
                squares=squares,
                origin="searched",
                method=method,
                seed=seed,
            )
        )
        promoted = bool(ok)
        if report.ok and report.L is not None:
            verified_L = float(report.L)

    elapsed = time.perf_counter() - t0
    improved_vs_baseline = (
        verified_L is not None and verified_L < float(baseline_L) - 1e-15
    )
    record_run(
        logs_dir,
        n=n,
        seed=seed,
        seed_type="phase4_best",
        L_verified=verified_L,
        improved=promoted,
        elapsed_s=elapsed,
        extras={
            "baseline_L": baseline_L,
            "improved_vs_baseline": improved_vs_baseline,
            "method": method,
            "budget_s": budget_s,
        },
    )
    return {
        "n": n,
        "seed": seed,
        "baseline_L": baseline_L,
        "verified_L": verified_L,
        "promoted": promoted,
        "improved_vs_baseline": improved_vs_baseline,
        "elapsed_s": elapsed,
        "method": method,
        "ok_fast": result.ok_fast,
        "steps": result.steps,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=_REPO)
    p.add_argument("--sample-mid", type=int, default=24, help="Sample size for n=51..324")
    p.add_argument("--sample-high", type=int, default=16, help="Sample size for n=325..1000")
    p.add_argument("--budget-mid-s", type=float, default=25.0)
    p.add_argument("--budget-high-s", type=float, default=20.0)
    p.add_argument("--seed0", type=int, default=20261109)
    p.add_argument("--no-polish", action="store_true")
    p.add_argument("--include-near-square", action="store_true")
    args = p.parse_args(argv)

    repo = args.repo_root.resolve()
    config = load_config(repo / "config.yaml")
    store = BestStore(config=config)
    if not store.best_dir.is_absolute():
        store.best_dir = repo / store.best_dir
    logs_dir = repo / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(repo / "results" / "baseline.csv")
    rng = np.random.default_rng(args.seed0)
    skip_near = not args.include_near_square

    mid_pool = candidate_ns(51, 324, skip_near)
    high_pool = candidate_ns(325, 1000, skip_near)
    mid = stratified_sample(mid_pool, args.sample_mid, rng)
    high = stratified_sample(high_pool, args.sample_high, rng)

    print(f"mid sample ({len(mid)}/{len(mid_pool)}): {mid}")
    print(f"high sample ({len(high)}/{len(high_pool)}): {high}")

    rows = []
    for n in mid:
        row = run_one(
            store,
            n,
            seed=args.seed0 + n,
            budget_s=args.budget_mid_s,
            logs_dir=logs_dir,
            baseline_L=float(baseline[n]["L"]),
            do_polish=not args.no_polish,
        )
        rows.append({**row, "range": "51-324"})
        print(
            f"  [mid] n={n} promoted={row['promoted']} improved_base={row['improved_vs_baseline']} "
            f"L={row['verified_L']} base={row['baseline_L']} t={row['elapsed_s']:.1f}s"
        )
    for n in high:
        row = run_one(
            store,
            n,
            seed=args.seed0 + n,
            budget_s=args.budget_high_s,
            logs_dir=logs_dir,
            baseline_L=float(baseline[n]["L"]),
            do_polish=not args.no_polish,
        )
        rows.append({**row, "range": "325-1000"})
        print(
            f"  [high] n={n} promoted={row['promoted']} improved_base={row['improved_vs_baseline']} "
            f"L={row['verified_L']} base={row['baseline_L']} t={row['elapsed_s']:.1f}s"
        )

    def summarize(label: str, subset: list[dict]) -> dict:
        return {
            "range": label,
            "sampled": len(subset),
            "promoted": sum(1 for r in subset if r["promoted"]),
            "improved_vs_baseline": sum(1 for r in subset if r["improved_vs_baseline"]),
            "mean_seconds": (sum(r["elapsed_s"] for r in subset) / max(len(subset), 1)),
            "ns_improved": [r["n"] for r in subset if r["improved_vs_baseline"]],
            "ns_promoted": [r["n"] for r in subset if r["promoted"]],
        }

    mid_rows = [r for r in rows if r["range"] == "51-324"]
    high_rows = [r for r in rows if r["range"] == "325-1000"]
    summary = {
        "seed0": args.seed0,
        "budgets_s": {"51-324": args.budget_mid_s, "325-1000": args.budget_high_s},
        "sample_sizes": {"51-324": len(mid), "325-1000": len(high)},
        "pool_sizes": {"51-324": len(mid_pool), "325-1000": len(high_pool)},
        "skip_near_square": skip_near,
        "polish": not args.no_polish,
        "ranges": [summarize("51-324", mid_rows), summarize("325-1000", high_rows)],
        "rows": rows,
    }
    out = logs_dir / "phase4_search_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    print(json.dumps(summary["ranges"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
