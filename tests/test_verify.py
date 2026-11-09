"""Verifier tests: fast filter, full verify, negative controls, shapely cross-check."""

from __future__ import annotations

import math

import numpy as np
import pytest

from squarepack.constructions.grid import grid_packing
from squarepack.geometry import sep_pair
from squarepack.verify import compute_L_verified, fast_filter, verify_full

shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon  # noqa: E402
from shapely import affinity  # noqa: E402


def _square_polygon(cx: float, cy: float, theta: float) -> Polygon:
    # Axis-aligned unit square centered at origin, then rotate and translate.
    poly = Polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    poly = affinity.rotate(poly, theta, origin=(0.0, 0.0), use_radians=True)
    return affinity.translate(poly, xoff=cx, yoff=cy)


def test_grid_n1_full_verify():
    squares = grid_packing(1)
    report, out = verify_full(squares)
    assert report.ok
    assert report.L == pytest.approx(1.0)
    assert report.delta == 0.0
    assert report.min_sep is None or report.min_sep == math.inf or report.min_sep >= 0.0
    assert report.min_margin is not None and report.min_margin >= 0.0
    assert out.shape == (1, 3)


def test_grid_n2_full_verify():
    report, _ = verify_full(grid_packing(2))
    assert report.ok
    assert report.L == pytest.approx(2.0)
    assert report.float64_ok and report.mp50_ok


def test_grid_n17_full_verify():
    report, _ = verify_full(grid_packing(17))
    assert report.ok
    assert report.L == pytest.approx(5.0)  # ceil(sqrt(17))=5


def test_fast_filter_accepts_grid():
    squares = grid_packing(10)
    r = fast_filter(squares, L=4.0)
    assert r.ok


def test_negative_control_overlap_rejected():
    """Take a verified packing, move one square so it overlaps, assert rejection.

    Spec §4.6: overlap by ~1e-9 must be rejected when dilation cannot repair it.
    With delta=0 the 1e-9 overlap fails immediately. With the default delta search
    (max 1e-6), a deeper overlap (1e-4) still fails because it exceeds ~delta_max*L.
    """
    squares = grid_packing(4)
    report, verified = verify_full(squares)
    assert report.ok

    bad_tiny = verified.copy()
    bad_tiny[1, 0] -= 1e-9
    fail_no_dilate, _ = verify_full(bad_tiny, delta=0.0)
    assert not fail_no_dilate.ok

    bad_deep = verified.copy()
    bad_deep[1, 0] -= 1e-4
    fail_default, _ = verify_full(bad_deep)
    assert not fail_default.ok


def test_negative_control_fast_filter_rejects_forced_small_L():
    """Containment against a too-small fixed L is rejected by the fast filter.

    Full verify recomputes L from corners (section 4.4), so self-derived L always
    contains the packing after normalization; the filter path is what takes an
    external container side.
    """
    squares = grid_packing(4)
    r = fast_filter(squares, L=1.5)  # true grid L is 2
    assert not r.ok
    assert r.min_margin is not None and r.min_margin < 0


def test_compute_L_rounds_up():
    squares = grid_packing(1)
    L = compute_L_verified(squares)
    assert L >= 1.0
    assert L == pytest.approx(1.0)


def test_shapely_cross_check_100k_pairs():
    """sep >= 0 iff shapely intersection area <= 1e-12, on 100k random pairs."""
    rng = np.random.default_rng(20261109)
    n_pairs = 100_000
    disagreements = 0
    for _ in range(n_pairs):
        c1x, c1y = rng.uniform(-2, 2, size=2)
        c2x, c2y = rng.uniform(-2, 2, size=2)
        t1 = float(rng.uniform(0, math.pi / 2))
        t2 = float(rng.uniform(0, math.pi / 2))
        s = sep_pair(c1x, c1y, t1, c2x, c2y, t2)
        p1 = _square_polygon(c1x, c1y, t1)
        p2 = _square_polygon(c2x, c2y, t2)
        area = p1.intersection(p2).area
        sep_nonneg = s >= 0.0
        shapely_disjoint = area <= 1e-12
        if sep_nonneg != shapely_disjoint:
            disagreements += 1

    assert disagreements == 0, f"{disagreements} disagreements out of {n_pairs}"


def test_sep_near_touching_matches_shapely():
    # Nearly touching pair along x
    s = sep_pair(0.5, 0.5, 0.0, 1.5 + 1e-8, 0.5, 0.0)
    assert s > 0
    p1 = _square_polygon(0.5, 0.5, 0.0)
    p2 = _square_polygon(1.5 + 1e-8, 0.5, 0.0)
    assert p1.intersection(p2).area <= 1e-12
