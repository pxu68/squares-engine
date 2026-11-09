"""Best-result store: promote only after full verify and improved L."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from squarepack.constructions.grid import grid_packing
from squarepack.store import BestStore, Candidate


@pytest.fixture
def store(tmp_path: Path) -> BestStore:
    return BestStore(best_dir=tmp_path / "best", config={
        "store": {"history_keep": 5},
        "verification": {
            "delta_default": 1e-12,
            "delta_max": 1e-6,
            "delta_growth": 10.0,
            "normalize": True,
            "mpmath_dps": 50,
        },
        "geometry": {"contact_sep_threshold": 1e-6},
    })


def test_promote_grid_writes_verified_json(store: BestStore):
    cand = Candidate(n=4, squares=grid_packing(4), origin="constructed", method="grid")
    ok, report, record = store.promote(cand)
    assert ok
    assert record is not None
    assert report.ok
    path = store.best_dir / "n0004.json"
    assert path.is_file()
    data = json.loads(path.read_text())
    assert data["n"] == 4
    assert data["L"] == pytest.approx(2.0)
    assert data["origin"] == "constructed"
    assert data["method"] == "grid"
    assert data["verified"]["float64"] is True
    assert data["verified"]["mp50"] is True
    assert len(data["squares"]) == 4


def test_promote_rejects_invalid(store: BestStore):
    squares = grid_packing(2)
    squares[1, 0] = squares[0, 0]  # total overlap
    cand = Candidate(n=2, squares=squares, origin="constructed", method="bad")
    ok, report, record = store.promote(cand)
    assert not ok
    assert record is None
    assert not report.ok
    assert store.load(2) is None


def test_promote_requires_strictly_improved_L(store: BestStore):
    cand = Candidate(n=3, squares=grid_packing(3), origin="constructed", method="grid")
    ok1, _, rec1 = store.promote(cand)
    assert ok1 and rec1 is not None
    # Same packing again: not an improvement
    ok2, report2, rec2 = store.promote(cand)
    assert not ok2
    assert rec2 is None
    assert "not improved" in report2.reason


def test_history_keeps_previous(store: BestStore):
    # Promote a deliberately larger valid packing for n=2 (L=3), then grid (L=2).
    wide = np.array([[0.5, 0.5, 0.0], [2.5, 0.5, 0.0]], dtype=np.float64)
    ok_w, _, rec_w = store.promote(
        Candidate(n=2, squares=wide, origin="constructed", method="wide")
    )
    assert ok_w and rec_w is not None
    assert rec_w["L"] == pytest.approx(3.0)
    ok_g, _, rec_g = store.promote(
        Candidate(n=2, squares=grid_packing(2), origin="constructed", method="grid")
    )
    assert ok_g and rec_g is not None
    assert rec_g["L"] == pytest.approx(2.0)
    hpath = store.best_dir / "n0002.history.json"
    assert hpath.is_file()
    hist = json.loads(hpath.read_text())
    assert hist["n"] == 2
    assert len(hist["history"]) >= 1
    assert hist["history"][-1]["method"] == "wide"
