"""Atomic best-result store (section 10.2)."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from squarepack.verify import VerifyReport, load_config_verify_kwargs, verify_full


def default_config_path() -> Path:
    # Prefer repo-root config.yaml relative to CWD, then package parent.
    cwd = Path.cwd() / "config.yaml"
    if cwd.is_file():
        return cwd
    here = Path(__file__).resolve()
    # src/squarepack/store.py -> repo root is parents[2]
    candidate = here.parents[2] / "config.yaml"
    return candidate


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else default_config_path()
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def result_path(best_dir: Path, n: int) -> Path:
    return best_dir / f"n{n:04d}.json"


def history_path(best_dir: Path, n: int) -> Path:
    return best_dir / f"n{n:04d}.history.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        # Best-effort directory fsync for durability on POSIX.
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass
class Candidate:
    """A packing candidate awaiting promotion."""

    n: int
    squares: np.ndarray
    origin: str
    method: str
    seed: int | None = None
    source: dict[str, Any] | None = None
    delta_hint: float | None = None


class BestStore:
    """results/best/nNNNN.json with promote-only-after-full-verify."""

    def __init__(
        self,
        best_dir: Path | str | None = None,
        *,
        config: dict[str, Any] | None = None,
        history_keep: int | None = None,
    ) -> None:
        cfg = config if config is not None else load_config()
        self.config = cfg
        paths = cfg.get("paths", {})
        if best_dir is None:
            best_dir = paths.get("results_best", "results/best")
        self.best_dir = Path(best_dir)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        store_cfg = cfg.get("store", {})
        self.history_keep = (
            int(history_keep)
            if history_keep is not None
            else int(store_cfg.get("history_keep", 5))
        )
        self.verify_kwargs = load_config_verify_kwargs(cfg)

    def load(self, n: int) -> dict[str, Any] | None:
        path = result_path(self.best_dir, n)
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def current_L(self, n: int) -> float | None:
        data = self.load(n)
        if data is None:
            return None
        return float(data["L"])

    def _append_history(self, n: int, record: dict[str, Any]) -> None:
        hpath = history_path(self.best_dir, n)
        history: list[dict[str, Any]] = []
        if hpath.is_file():
            with hpath.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                history = raw
            elif isinstance(raw, dict) and isinstance(raw.get("history"), list):
                history = list(raw["history"])
        history.append(record)
        history = history[-self.history_keep :]
        _atomic_write_json(hpath, {"n": n, "history": history})

    def promote(self, candidate: Candidate) -> tuple[bool, VerifyReport, dict[str, Any] | None]:
        """Run full verification; replace stored best only if verified L improves.

        Returns (promoted, report, stored_record_or_None).
        """
        squares = np.asarray(candidate.squares, dtype=np.float64)
        if squares.shape[0] != candidate.n:
            raise ValueError(f"n={candidate.n} but got {squares.shape[0]} squares")

        kwargs = dict(self.verify_kwargs)
        if candidate.delta_hint is not None:
            kwargs["delta"] = float(candidate.delta_hint)

        report, verified_squares = verify_full(squares, **kwargs)
        if not report.ok or report.L is None:
            return False, report, None

        report.extras["verified_squares"] = verified_squares

        record = {
            "n": candidate.n,
            "L": float(report.L),
            "delta": float(report.delta),
            "origin": candidate.origin,
            "method": candidate.method,
            "seed": candidate.seed,
            "source": candidate.source
            or {"finder": None, "date": None, "url": None},
            "verified": {
                "float64": bool(report.float64_ok),
                "mp50": bool(report.mp50_ok),
                "min_sep": report.min_sep,
                "min_margin": report.min_margin,
            },
            "squares": [
                [float(x), float(y), float(t)] for x, y, t in verified_squares.tolist()
            ],
        }

        current = self.load(candidate.n)
        if current is not None and float(current["L"]) <= float(report.L):
            # Not an improvement (strictly smaller L required).
            report.reason = (
                f"verified but not improved: new L={report.L} vs stored L={current['L']}"
            )
            return False, report, record

        # Keep previous best in history before replacing.
        if current is not None:
            self._append_history(candidate.n, current)

        _atomic_write_json(result_path(self.best_dir, candidate.n), record)
        return True, report, record
