#!/usr/bin/env python3
"""Phase 3 acceptance: reproduce snapshot/best L for n=2..30; bench a few n in 30..60."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from squarepack.search.anneal import anneal  # noqa: E402
from squarepack.search.basin import record_run  # noqa: E402
from squarepack.store import BestStore, Candidate, load_config  # noqa: E402


def load_snapshot_L(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("verified_ok", "")).lower() not in ("true", "1", "yes"):
                continue
            out[int(float(row["n"]))] = float(row["L_verified"])
    return out


def target_for_n(n: int, store: BestStore, snap: dict[int, float]) -> float:
    if n in snap:
        return float(snap[n])
    cur = store.current_L(n)
    if cur is None:
        raise RuntimeError(f"no target L for n={n}")
    return float(cur)


def reproduce_one(
    store: BestStore,
    n: int,
    target: float,
    *,
    seed: int,
    budget_s: float,
    logs_dir: Path,
    tol: float = 1e-6,
    promote: bool = False,
) -> dict:
    rec = store.load(n)
    init = None if rec is None else np.asarray(rec["squares"], dtype=np.float64)
    t0 = time.perf_counter()
    result = anneal(
        n,
        seed,
        init_squares=init,
        best_known_L=target,
        time_budget_s=budget_s,
        temperature=1e-4,  # low-temp seeded reproduction
        eta0=5e-3,
        early_stop_L=target,
        early_stop_tol=tol,
    )
    verified_L = None
    ok_promote = False
    if result.ok_fast:
        ok, report, _ = store.promote(
            Candidate(
                n=n,
                squares=result.squares,
                origin="searched",
                method="anneal_repro",
                seed=seed,
            )
        ) if promote else (False, None, None)
        if promote and report is not None and report.ok and report.L is not None:
            verified_L = float(report.L)
            ok_promote = bool(ok)
        else:
            # Verify without requiring improvement over best
            from squarepack.verify import verify_full

            report2, _ = verify_full(result.squares)
            if report2.ok and report2.L is not None:
                verified_L = float(report2.L)
    elapsed = time.perf_counter() - t0
    hit = verified_L is not None and verified_L <= target + tol
    record_run(
        logs_dir,
        n=n,
        seed=seed,
        seed_type="repro_best",
        L_verified=verified_L,
        improved=ok_promote,
        elapsed_s=elapsed,
        extras={"target": target, "hit": hit, "fast_L": result.L},
    )
    return {
        "n": n,
        "target": target,
        "verified_L": verified_L,
        "fast_L": result.L,
        "hit": hit,
        "gap": None if verified_L is None else verified_L - target,
        "elapsed_s": elapsed,
        "steps": result.steps,
        "ok_fast": result.ok_fast,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=_REPO)
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--n-max", type=int, default=30)
    p.add_argument("--budget-s", type=float, default=20.0, help="Per-n reproduction budget")
    p.add_argument("--bench-ns", type=str, default="31,37,41,47,53", help="Comma n in 30..60")
    p.add_argument("--bench-budget-s", type=float, default=60.0)
    p.add_argument("--bench-seeds", type=int, default=3)
    p.add_argument("--seed0", type=int, default=20261109)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--promote", action="store_true")
    args = p.parse_args(argv)

    repo = args.repo_root.resolve()
    config = load_config(repo / "config.yaml")
    store = BestStore(config=config)
    logs_dir = repo / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    snap = load_snapshot_L(repo / "data" / "records_snapshot.csv")

    repro_rows = []
    print("== reproduction n=2..30 ==")
    for n in range(args.n_min, args.n_max + 1):
        target = target_for_n(n, store, snap)
        row = reproduce_one(
            store,
            n,
            target,
            seed=args.seed0 + n,
            budget_s=args.budget_s,
            logs_dir=logs_dir,
            tol=args.tol,
            promote=args.promote,
        )
        repro_rows.append(row)
        status = "HIT" if row["hit"] else "MISS"
        print(
            f"  n={n:3d} target={target:.9f} verified={row['verified_L']} "
            f"gap={row['gap']} {status} t={row['elapsed_s']:.1f}s"
        )

    hits = sum(1 for r in repro_rows if r["hit"])
    print(f"reproduction hits={hits}/{len(repro_rows)}")

    bench = []
    if not args.skip_bench:
        print("== bench n in 30..60 ==")
        ns = [int(x) for x in args.bench_ns.split(",") if x.strip()]
        for n in ns:
            target = target_for_n(n, store, snap)
            hits_n = 0
            times = []
            for k in range(args.bench_seeds):
                seed = args.seed0 + 10_000 * n + k
                row = reproduce_one(
                    store,
                    n,
                    target,
                    seed=seed,
                    budget_s=args.bench_budget_s,
                    logs_dir=logs_dir,
                    tol=args.tol,
                    promote=False,
                )
                times.append(row["elapsed_s"])
                if row["hit"]:
                    hits_n += 1
                print(
                    f"  n={n} seed={seed} hit={row['hit']} verified={row['verified_L']} "
                    f"t={row['elapsed_s']:.1f}s"
                )
            hit_rate = hits_n / max(args.bench_seeds, 1)
            mean_t = sum(times) / max(len(times), 1)
            core_hours_est = mean_t / 3600.0  # single-core seconds -> hours
            bench.append(
                {
                    "n": n,
                    "target": target,
                    "seeds": args.bench_seeds,
                    "hits": hits_n,
                    "hit_rate": hit_rate,
                    "mean_seconds": mean_t,
                    "core_hours_per_seed": core_hours_est,
                    "budget_s": args.bench_budget_s,
                }
            )
            print(f"  summary n={n} hit_rate={hit_rate:.3f} mean_s={mean_t:.1f}")

    out = {
        "reproduction": {
            "n_min": args.n_min,
            "n_max": args.n_max,
            "tol": args.tol,
            "budget_s": args.budget_s,
            "hits": hits,
            "total": len(repro_rows),
            "rows": repro_rows,
        },
        "bench": bench,
    }
    out_path = logs_dir / "phase3_acceptance.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_path)
    return 0 if hits == len(repro_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
