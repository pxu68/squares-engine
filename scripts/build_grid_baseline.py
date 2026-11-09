#!/usr/bin/env python3
"""Build verified axis-aligned grid packings for n = 1..N into results/best/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install: add src/ to path.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from squarepack.constructions.grid import grid_claimed_L, grid_packing  # noqa: E402
from squarepack.store import BestStore, Candidate, load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=1000)
    parser.add_argument(
        "--best-dir",
        type=Path,
        default=None,
        help="Override results/best directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO / "config.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-promote even if a result already exists (still requires improved L)",
    )
    parser.add_argument(
        "--overwrite-clear",
        action="store_true",
        help="Delete existing best JSON for each n before promoting (grid rebuild)",
    )
    args = parser.parse_args(argv)

    if args.n_min < 1 or args.n_max < args.n_min:
        print("invalid n range", file=sys.stderr)
        return 2

    config = load_config(args.config)
    store = BestStore(best_dir=args.best_dir, config=config)
    grid_cfg = config.get("grid", {})
    origin = str(grid_cfg.get("origin", "constructed"))
    method = str(grid_cfg.get("method", "grid"))

    promoted = 0
    skipped = 0
    failed = 0

    for n in range(args.n_min, args.n_max + 1):
        path = store.best_dir / f"n{n:04d}.json"
        if args.overwrite_clear and path.is_file():
            path.unlink()

        if path.is_file() and not args.force and not args.overwrite_clear:
            skipped += 1
            continue

        squares = grid_packing(n)
        cand = Candidate(
            n=n,
            squares=squares,
            origin=origin,
            method=method,
            seed=None,
            source={"finder": None, "date": None, "url": None},
        )
        ok, report, record = store.promote(cand)
        if ok:
            promoted += 1
            assert record is not None
            claimed = grid_claimed_L(n)
            print(
                f"n={n:4d}  L={record['L']:.17g}  claimed_k={claimed}  "
                f"delta={record['delta']:.3g}  min_sep={report.min_sep}"
            )
        else:
            # If already present with equal/better L, count as skipped.
            if store.load(n) is not None and "not improved" in report.reason:
                skipped += 1
            else:
                failed += 1
                print(f"n={n:4d} FAILED: {report.reason}", file=sys.stderr)

    print(
        f"done: promoted={promoted} skipped={skipped} failed={failed} "
        f"range={args.n_min}..{args.n_max} best_dir={store.best_dir}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
