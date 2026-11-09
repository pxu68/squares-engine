"""Grid construction tests."""

from __future__ import annotations

import math

import pytest

from squarepack.constructions.grid import grid_claimed_L, grid_k, grid_packing
from squarepack.verify import verify_full


@pytest.mark.parametrize("n,k", [(1, 1), (2, 2), (3, 2), (4, 2), (5, 3), (100, 10), (1000, 32)])
def test_grid_k(n: int, k: int):
    assert grid_k(n) == k
    assert grid_claimed_L(n) == float(k)


def test_grid_packing_shape_and_bounds():
    squares = grid_packing(10)
    assert squares.shape == (10, 3)
    assert (squares[:, 2] == 0.0).all()
    k = 4
    assert squares[:, 0].min() >= 0.5
    assert squares[:, 0].max() <= k - 0.5
    assert squares[:, 1].min() >= 0.5
    assert squares[:, 1].max() <= k - 0.5


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 100])
def test_grid_verifies(n: int):
    report, out = verify_full(grid_packing(n))
    assert report.ok, report.reason
    assert report.L == pytest.approx(float(math.ceil(math.sqrt(n))))
    assert out.shape == (n, 3)
