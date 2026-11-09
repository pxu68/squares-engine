"""Tests for Ellsworth SVG parsing (uses cached fixtures when present)."""

from __future__ import annotations

from pathlib import Path

import pytest

from squarepack.constructions.grid import grid_claimed_L
from squarepack.records.svg_to_squares import svg_to_packing
from squarepack.verify import verify_full

FIXTURE_DIR = Path("/tmp/sqpack_probe")


@pytest.mark.parametrize(
    "name,n",
    [
        ("square-5.svg", 5),
        ("square-10.svg", 10),
        ("square-11.svg", 11),
        ("square-17.svg", 17),
    ],
)
def test_parse_and_verify_fixture(name: str, n: int):
    path = FIXTURE_DIR / name
    if not path.is_file():
        pytest.skip(f"fixture {path} not available")
    squares, L_pub, meta = svg_to_packing(path)
    assert squares.shape == (n, 3)
    assert meta["n_parsed"] == n
    report, _ = verify_full(squares)
    assert report.ok, report.reason
    assert report.L is not None
    assert report.L < grid_claimed_L(n) or abs(report.L - L_pub) < 1e-6
