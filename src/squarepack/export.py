"""Export packings (section 10.3).

Default format (placeholder until an official grader is supplied): CSV with
header ``n,i,x,y,theta`` plus ``L_by_n.csv`` with ``n,L``.

Owner-confirmed: L / container / CSV are placeholders; imported records are
seeds only — use ``--exclude-origin imported``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from squarepack.store import BestStore, load_config
from squarepack.verify import verify_full


def _parse_n_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def select_ns(
    store: BestStore,
    *,
    all_n: bool = False,
    n_list: Sequence[int] | None = None,
    n_range: tuple[int, int] | None = None,
    exclude_origins: Iterable[str] | None = None,
) -> list[int]:
    exclude = {o.strip() for o in (exclude_origins or []) if o.strip()}
    if all_n:
        # Discover from disk
        ns = sorted(
            int(p.stem[1:])
            for p in store.best_dir.glob("n????.json")
            if p.stem.startswith("n") and p.stem[1:].isdigit()
        )
    elif n_list:
        ns = list(n_list)
    elif n_range:
        lo, hi = n_range
        ns = list(range(int(lo), int(hi) + 1))
    else:
        raise ValueError("select one of --all, --n, or --range")

    selected: list[int] = []
    for n in ns:
        rec = store.load(n)
        if rec is None:
            continue
        if rec.get("origin") in exclude:
            continue
        selected.append(n)
    return selected


def write_export(
    store: BestStore,
    ns: Sequence[int],
    out_dir: Path,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Write squares.csv and L_by_n.csv; return paths and record summaries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    squares_path = out_dir / "squares.csv"
    L_path = out_dir / "L_by_n.csv"
    summaries: list[dict[str, Any]] = []

    with squares_path.open("w", encoding="utf-8", newline="") as sf, L_path.open(
        "w", encoding="utf-8", newline=""
    ) as lf:
        sw = csv.DictWriter(sf, fieldnames=["n", "i", "x", "y", "theta"])
        lw = csv.DictWriter(lf, fieldnames=["n", "L", "origin", "method"])
        sw.writeheader()
        lw.writeheader()
        for n in ns:
            rec = store.load(n)
            if rec is None:
                raise FileNotFoundError(f"missing results/best for n={n}")
            squares = rec["squares"]
            for i, row in enumerate(squares):
                sw.writerow(
                    {
                        "n": n,
                        "i": i,
                        "x": repr(float(row[0])),
                        "y": repr(float(row[1])),
                        "theta": repr(float(row[2])),
                    }
                )
            lw.writerow(
                {
                    "n": n,
                    "L": repr(float(rec["L"])),
                    "origin": rec.get("origin", ""),
                    "method": rec.get("method", ""),
                }
            )
            summaries.append(
                {
                    "n": n,
                    "L": float(rec["L"]),
                    "origin": rec.get("origin"),
                    "method": rec.get("method"),
                    "n_squares": len(squares),
                }
            )
    return squares_path, L_path, summaries


def load_packings_from_export(squares_csv: Path) -> dict[int, np.ndarray]:
    """Rebuild packings from squares.csv alone (section 10.3 self-check)."""
    buckets: dict[int, list[tuple[int, float, float, float]]] = {}
    with squares_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n = int(row["n"])
            i = int(row["i"])
            buckets.setdefault(n, []).append(
                (i, float(row["x"]), float(row["y"]), float(row["theta"]))
            )
    out: dict[int, np.ndarray] = {}
    for n, rows in buckets.items():
        rows_sorted = sorted(rows, key=lambda t: t[0])
        arr = np.array([[x, y, th] for _, x, y, th in rows_sorted], dtype=np.float64)
        out[n] = arr
    return out


def reverify_export(
    squares_csv: Path,
    *,
    verify_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-verify every packing from the exported file alone."""
    packings = load_packings_from_export(squares_csv)
    failures: list[dict[str, Any]] = []
    ok_n = 0
    for n in sorted(packings):
        report, _ = verify_full(packings[n], **(verify_kwargs or {}))
        if report.ok:
            ok_n += 1
        else:
            failures.append({"n": n, "reason": report.reason})
    return {
        "n_packings": len(packings),
        "ok": ok_n,
        "failures": failures,
        "passed": len(failures) == 0 and ok_n == len(packings),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true", help="Export every stored n")
    sel.add_argument("--n", type=str, help="Comma list, e.g. 1,5,17")
    sel.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("LO", "HI"),
        help="Inclusive n range",
    )
    p.add_argument(
        "--exclude-origin",
        action="append",
        default=[],
        help="Origin to exclude (repeatable). Prefer: imported",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/export/<stamp>)",
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        default=True,
        help="Re-verify exported packings from the CSV (default on)",
    )
    p.add_argument("--no-self-check", action="store_true")
    args = p.parse_args(argv)

    repo = args.repo_root.resolve()
    config = load_config(repo / "config.yaml")
    store = BestStore(config=config)
    # Ensure best_dir is under repo when relative
    if not store.best_dir.is_absolute():
        store.best_dir = repo / store.best_dir

    exclude = list(args.exclude_origin)
    ns = select_ns(
        store,
        all_n=bool(args.all),
        n_list=_parse_n_list(args.n) if args.n else None,
        n_range=(args.range[0], args.range[1]) if args.range else None,
        exclude_origins=exclude,
    )
    if not ns:
        print("no packings selected", file=sys.stderr)
        return 2

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = repo / "results" / "export" / "latest"
    out_dir = out_dir if out_dir.is_absolute() else repo / out_dir

    squares_path, L_path, summaries = write_export(store, ns, out_dir)
    meta = {
        "n_exported": len(ns),
        "exclude_origin": exclude,
        "squares_csv": str(squares_path),
        "L_by_n_csv": str(L_path),
        "summaries": summaries,
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"exported {len(ns)} packings -> {squares_path}")
    print(f"L table -> {L_path}")

    do_check = args.self_check and not args.no_self_check
    if do_check:
        check = reverify_export(squares_path, verify_kwargs=store.verify_kwargs)
        (out_dir / "export_reverify.json").write_text(
            json.dumps(check, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"self-check passed={check['passed']} ok={check['ok']}/{check['n_packings']} "
            f"failures={len(check['failures'])}"
        )
        if not check["passed"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
