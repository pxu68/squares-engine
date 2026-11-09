#!/usr/bin/env python3
"""Build verified baselines for n=1..1000 over imports + constructions (Phase 2)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from squarepack.constructions.composition import composition_candidates_for_n  # noqa: E402
from squarepack.constructions.goebel_strip import goebel_candidates_for_n  # noqa: E402
from squarepack.constructions.grid import grid_claimed_L, grid_packing  # noqa: E402
from squarepack.constructions.l_extension import l_extension  # noqa: E402
from squarepack.constructions.neighbor_seed import seed_from_larger, seed_insert_one  # noqa: E402
from squarepack.constructions.removal import remove_to_n  # noqa: E402
from squarepack.records.import_records import import_records  # noqa: E402
from squarepack.store import BestStore, Candidate, load_config  # noqa: E402
from squarepack.verify import fast_filter  # noqa: E402


def _load_best_maps(store: BestStore, n_max: int) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, dict]]:
    squares: dict[int, np.ndarray] = {}
    Ls: dict[int, float] = {}
    meta: dict[int, dict] = {}
    for n in range(1, n_max + 1):
        rec = store.load(n)
        if rec is None:
            continue
        squares[n] = np.asarray(rec["squares"], dtype=np.float64)
        Ls[n] = float(rec["L"])
        meta[n] = rec
    return squares, Ls, meta


def _try_promote(
    store: BestStore,
    n: int,
    squares: np.ndarray,
    *,
    origin: str,
    method: str,
) -> bool:
    # Cheap reject before full verify/promote.
    ff = fast_filter(squares)
    if not ff.ok:
        return False
    cur = store.current_L(n)
    if cur is not None and ff.L is not None and ff.L >= cur - 1e-15:
        return False
    ok, report, _ = store.promote(
        Candidate(n=n, squares=squares, origin=origin, method=method, seed=None)
    )
    if ok:
        print(f"  promote n={n} L={report.L} method={method}")
    return ok


def ensure_grid(store: BestStore, n_min: int, n_max: int) -> int:
    promoted = 0
    for n in range(n_min, n_max + 1):
        if store.load(n) is not None:
            continue
        if _try_promote(
            store, n, grid_packing(n), origin="constructed", method="grid"
        ):
            promoted += 1
    return promoted


def apply_l_extensions(store: BestStore, n_max: int, passes: int = 3) -> int:
    promoted = 0
    for _ in range(passes):
        squares, Ls, _ = _load_best_maps(store, n_max)
        gained = 0
        for m in sorted(squares):
            s = Ls[m]
            ext = l_extension(squares[m], s)
            if ext is None:
                continue
            new_squares, new_n = ext
            if new_n > n_max:
                continue
            if _try_promote(
                store,
                new_n,
                new_squares,
                origin="constructed",
                method=f"l_extension_from_{m}",
            ):
                gained += 1
                promoted += 1
        if gained == 0:
            break
    return promoted


def apply_removal(store: BestStore, n_max: int, max_source_gap: int = 80) -> int:
    promoted = 0
    squares, Ls, _ = _load_best_maps(store, n_max)
    # Prefer sources that beat the grid — removal preserves geometry and may shrink L.
    sources = sorted(
        (m for m in squares if Ls[m] < grid_claimed_L(m) - 1e-15),
        key=lambda m: Ls[m] / math.sqrt(m),
    )
    # Also include a few perfect-grid sources near each n for completeness (cheap).
    for n in range(1, n_max + 1):
        current_L = Ls.get(n)
        best_cand = None
        best_method = None
        # From improved sources
        for m in sources:
            if m <= n or m > n + max_source_gap:
                continue
            # Only useful if source L is below current / grid
            target = current_L if current_L is not None else grid_claimed_L(n)
            if Ls[m] >= target - 1e-15:
                continue
            cand = remove_to_n(squares[m], n, Ls[m])
            if cand is None:
                continue
            best_cand = cand
            best_method = f"removal_from_{m}"
            # Try immediately; first improving source for this n is enough this pass.
            if _try_promote(
                store, n, best_cand, origin="constructed", method=best_method
            ):
                promoted += 1
                current_L = store.current_L(n)
                squares[n] = np.asarray(store.load(n)["squares"], dtype=np.float64)
                Ls[n] = float(current_L)  # type: ignore[arg-type]
            # continue scanning other sources for possible further improvement
    return promoted


def apply_goebel(store: BestStore, n_max: int, n_cap: int = 120) -> int:
    promoted = 0
    for n in range(1, min(n_max, n_cap) + 1):
        cur = store.current_L(n)
        grid_L = grid_claimed_L(n)
        if cur is not None and cur < grid_L - 1e-9:
            continue  # already beaten
        for cand in goebel_candidates_for_n(n, max_strip=10, max_rows=6, limit=20):
            if _try_promote(
                store, n, cand, origin="constructed", method="goebel_strip"
            ):
                promoted += 1
                break
    return promoted


def apply_composition(store: BestStore, n_max: int, n_cap: int = 120) -> int:
    promoted = 0
    for n in range(1, min(n_max, n_cap) + 1):
        cur = store.current_L(n)
        grid_L = grid_claimed_L(n)
        if cur is not None and cur < grid_L - 1e-9:
            continue
        for cand in composition_candidates_for_n(n, max_side=7, limit=20):
            if _try_promote(
                store, n, cand, origin="constructed", method="corner_composition"
            ):
                promoted += 1
                break
    return promoted


def apply_neighbor_seeds(store: BestStore, n_max: int) -> int:
    """Geometric neighbor seeds without section-7 relaxation."""
    promoted = 0
    squares, Ls, _ = _load_best_maps(store, n_max)
    for n in range(1, n_max + 1):
        # from n+1
        if n + 1 in squares:
            cand = seed_from_larger(squares[n + 1], n)
            if cand is not None:
                if _try_promote(
                    store, n, cand, origin="constructed", method="neighbor_from_nplus1"
                ):
                    promoted += 1
                    squares, Ls, _ = _load_best_maps(store, n_max)
        # insert into n-1
        if n - 1 in squares and n - 1 >= 1:
            cand = seed_insert_one(squares[n - 1], Ls[n - 1])
            if cand is not None and cand.shape[0] == n:
                if _try_promote(
                    store, n, cand, origin="constructed", method="neighbor_insert"
                ):
                    promoted += 1
                    squares, Ls, _ = _load_best_maps(store, n_max)
    return promoted


def write_baseline_csv(store: BestStore, path: Path, n_max: int) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    beat_grid = 0
    imported = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["n", "L", "origin", "method", "sqrt_n", "gap"]
        )
        w.writeheader()
        for n in range(1, n_max + 1):
            rec = store.load(n)
            if rec is None:
                raise RuntimeError(f"missing best result for n={n}")
            L = float(rec["L"])
            sqrt_n = math.sqrt(n)
            gap = L - sqrt_n
            origin = rec.get("origin", "")
            method = rec.get("method", "")
            if origin == "imported":
                imported += 1
            if L < grid_claimed_L(n) - 1e-15:
                beat_grid += 1
            w.writerow(
                {
                    "n": n,
                    "L": L,
                    "origin": origin,
                    "method": method,
                    "sqrt_n": sqrt_n,
                    "gap": gap,
                }
            )
    return {"beat_grid": beat_grid, "imported_in_best": imported}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_REPO)
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=1000)
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--skip-goebel", action="store_true")
    parser.add_argument("--skip-composition", action="store_true")
    parser.add_argument("--skip-neighbor", action="store_true")
    parser.add_argument("--construction-cap", type=int, default=150,
                        help="Max n for expensive Goebel/composition scans")
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve()
    config = load_config(repo / "config.yaml")
    store = BestStore(config=config)

    print("== ensure grid ==")
    g = ensure_grid(store, args.n_min, args.n_max)
    print(f"grid promoted={g}")

    import_report = None
    if not args.skip_import:
        print("== import records ==")
        import_report = import_records(
            repo_root=repo,
            n_min=args.n_min,
            n_max=args.n_max,
            promote=True,
        )
        print(
            f"import imported={import_report['imported']} verified={import_report['verified']} "
            f"failures={import_report['failures']} promoted={import_report['promoted']} "
            f"within_1e9={import_report['within_1e9']}"
        )
        if import_report.get("fatal"):
            print("IMPORT FATAL:", import_report["fatal"], file=sys.stderr)

    print("== L-extensions ==")
    print("l_extension promoted=", apply_l_extensions(store, args.n_max))

    print("== removal ==")
    print("removal promoted=", apply_removal(store, args.n_max))

    if not args.skip_goebel:
        print("== goebel ==")
        print(
            "goebel promoted=",
            apply_goebel(store, args.n_max, n_cap=args.construction_cap),
        )

    if not args.skip_composition:
        print("== composition ==")
        print(
            "composition promoted=",
            apply_composition(store, args.n_max, n_cap=args.construction_cap),
        )

    # Another removal/extension pass after new constructions.
    print("== L-extensions (pass 2) ==")
    print("l_extension promoted=", apply_l_extensions(store, args.n_max))
    print("== removal (pass 2) ==")
    print("removal promoted=", apply_removal(store, args.n_max))

    if not args.skip_neighbor:
        print("== neighbor seeds ==")
        print("neighbor promoted=", apply_neighbor_seeds(store, args.n_max))

    out_csv = repo / "results" / "baseline.csv"
    stats = write_baseline_csv(store, out_csv, args.n_max)
    summary = {
        "n_max": args.n_max,
        "baseline_csv": str(out_csv),
        "beat_grid": stats["beat_grid"],
        "imported_in_best": stats["imported_in_best"],
        "import_report": None
        if import_report is None
        else {
            k: import_report[k]
            for k in (
                "imported",
                "parsed",
                "verified",
                "promoted",
                "failures",
                "within_1e9",
                "snapshot_time",
                "fatal",
            )
            if k in import_report
        },
    }
    summary_path = repo / "results" / "baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("== baseline written ==")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
