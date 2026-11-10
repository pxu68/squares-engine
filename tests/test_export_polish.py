"""Tests for Phase 4 export and polish."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from squarepack.constructions.grid import grid_packing
from squarepack.export import reverify_export, select_ns, write_export
from squarepack.polish import polish
from squarepack.store import BestStore, Candidate


def test_polish_grid_runs(tmp_path: Path):
    sq = grid_packing(4)
    result = polish(sq)
    assert result.ok
    assert result.L is not None
    assert result.L <= 2.0 + 1e-9


def test_export_exclude_imported_and_reverify(tmp_path: Path):
    best = tmp_path / "best"
    best.mkdir()
    store = BestStore(best_dir=best, config={"store": {"history_keep": 2}, "verification": {}})
    # constructed
    ok, _, _ = store.promote(
        Candidate(n=4, squares=grid_packing(4), origin="constructed", method="grid")
    )
    assert ok
    # fake imported with same geometry
    ok2, _, rec = store.promote(
        Candidate(n=1, squares=grid_packing(1), origin="imported", method="ellsworth_svg")
    )
    assert ok2
    assert rec is not None

    ns_all = select_ns(store, all_n=True)
    assert set(ns_all) == {1, 4}
    ns_ex = select_ns(store, all_n=True, exclude_origins=["imported"])
    assert ns_ex == [4]

    out = tmp_path / "exp"
    squares_path, L_path, summaries = write_export(store, ns_ex, out)
    assert squares_path.is_file() and L_path.is_file()
    assert len(summaries) == 1 and summaries[0]["n"] == 4
    check = reverify_export(squares_path)
    assert check["passed"] is True
    assert check["ok"] == 1
