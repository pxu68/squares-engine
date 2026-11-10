"""Unit tests for Phase 3 energy / moves / anneal smoke."""

from __future__ import annotations

import math

import numpy as np

from squarepack.constructions.grid import grid_packing
from squarepack.search.anneal import anneal
from squarepack.search.energy import energy_and_grad
from squarepack.search.moves import apply_random_move, relocate


def test_grid_energy_near_zero_at_exact_container():
    sq = grid_packing(4)
    e, g = energy_and_grad(sq, s=2.0)
    assert e < 1e-20
    assert np.max(np.abs(g)) < 1e-10


def test_overlap_has_positive_energy():
    sq = np.array([[1.0, 1.0, 0.0], [1.1, 1.0, 0.0]], dtype=np.float64)
    e, g = energy_and_grad(sq, s=3.0)
    assert e > 0.0
    assert np.any(g != 0.0)


def test_moves_preserve_count():
    rng = np.random.default_rng(0)
    sq = grid_packing(5)
    for _ in range(20):
        out, name = apply_random_move(sq, rng, s=3.0)
        assert out.shape == sq.shape
        assert name
    out = relocate(sq, rng, 3.0)
    assert out.shape == sq.shape


def test_anneal_smoke_preserves_grid():
    sq = grid_packing(4)
    res = anneal(4, seed=1, init_squares=sq, best_known_L=2.0, time_budget_s=2.0, temperature=1e-4)
    assert res.n == 4
    assert res.ok_fast
    assert res.L <= 2.0 + 1e-6
    assert math.isfinite(res.elapsed_s)
