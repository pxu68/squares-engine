#!/usr/bin/env python3
"""Enqueue round-robin search jobs and run the file-based scheduler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from squarepack.scheduler import (  # noqa: E402
    build_round_robin_jobs,
    enqueue,
    run_scheduler,
)
from squarepack.store import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=_REPO)
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--n-max", type=int, default=30)
    p.add_argument("--seeds-per-n", type=int, default=1)
    p.add_argument("--seed0", type=int, default=1)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--budget-s", type=float, default=None, help="Override per-job budget")
    p.add_argument("--enqueue-only", action="store_true")
    p.add_argument("--run-only", action="store_true")
    args = p.parse_args(argv)

    repo = args.repo_root.resolve()
    config = load_config(repo / "config.yaml")
    workers = args.workers if args.workers is not None else int(config.get("search", {}).get("workers", 1))
    logs_dir = repo / "results" / "logs"

    if not args.run_only:
        jobs = build_round_robin_jobs(
            args.n_min,
            args.n_max,
            seeds_per_n=args.seeds_per_n,
            seed0=args.seed0,
            skip_near_square=bool(config.get("search", {}).get("skip_near_square", True)),
            logs_dir=logs_dir,
            max_runs_deprioritize=int(config.get("search", {}).get("max_basin_runs_deprioritize", 200)),
        )
        n_new = enqueue(repo, jobs)
        print(f"enqueued {n_new} new jobs (total requested {len(jobs)})")

    if args.enqueue_only:
        return 0

    run_scheduler(repo, workers=workers, time_budget_s=args.budget_s)
    print("scheduler finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
