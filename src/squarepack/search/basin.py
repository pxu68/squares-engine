"""Basin statistics logging (section 8)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def basin_path(logs_dir: Path, n: int) -> Path:
    return Path(logs_dir) / f"basin_n{n}.json"


def load_basin(logs_dir: Path, n: int) -> dict[str, Any]:
    path = basin_path(logs_dir, n)
    if not path.is_file():
        return {
            "n": n,
            "runs": [],
            "histogram": {},
            "by_seed_type": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def record_run(
    logs_dir: Path,
    *,
    n: int,
    seed: int,
    seed_type: str,
    L_verified: float | None,
    improved: bool,
    elapsed_s: float,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one run and update histogram of verified L rounded to 1e-6."""
    data = load_basin(logs_dir, n)
    entry = {
        "seed": seed,
        "seed_type": seed_type,
        "L_verified": L_verified,
        "improved": improved,
        "elapsed_s": elapsed_s,
    }
    if extras:
        entry.update(extras)
    data["runs"].append(entry)

    if L_verified is not None and math.isfinite(L_verified):
        key = f"{round(L_verified, 6):.6f}"
        hist = data.setdefault("histogram", {})
        hist[key] = int(hist.get(key, 0)) + 1
        bst = data.setdefault("by_seed_type", {})
        bucket = bst.setdefault(seed_type, {"runs": 0, "improvements": 0, "histogram": {}})
        bucket["runs"] = int(bucket["runs"]) + 1
        if improved:
            bucket["improvements"] = int(bucket["improvements"]) + 1
        bucket["histogram"][key] = int(bucket["histogram"].get(key, 0)) + 1

    _atomic_write(basin_path(logs_dir, n), data)
    return data


def distinct_basins(data: dict[str, Any]) -> int:
    return len(data.get("histogram") or {})


def should_deprioritize(
    data: dict[str, Any],
    *,
    seed_type: str,
    max_runs: int = 200,
) -> bool:
    """Deprioritize after max_runs with no improvement and no new basin."""
    bst = (data.get("by_seed_type") or {}).get(seed_type) or {}
    runs = int(bst.get("runs", 0))
    improvements = int(bst.get("improvements", 0))
    basins = len(bst.get("histogram") or {})
    if runs < max_runs:
        return False
    return improvements == 0 and basins <= 1
