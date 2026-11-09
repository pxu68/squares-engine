"""Unit tests for geometry primitives."""

from __future__ import annotations

import math

import numpy as np
import pytest

from squarepack.geometry import (
    bounding_box,
    container_length_float64,
    corners,
    half_extent,
    sep_pair,
    support_radius,
)


def test_corners_axis_aligned():
    c = corners(0.5, 0.5, 0.0)
    expected = {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}
    got = {(round(x, 12), round(y, 12)) for x, y in c}
    assert got == expected


def test_support_radius_axis_aligned():
    # Along x-axis (phi=0), axis-aligned square has r = 0.5
    assert support_radius(0.0, 0.0) == pytest.approx(0.5)
    assert support_radius(math.pi / 2, 0.0) == pytest.approx(0.5)
    # 45-degree square along x-axis: r = 0.5*(|cos(-pi/4)|+|sin(-pi/4)|)=sqrt(2)/2
    assert support_radius(0.0, math.pi / 4) == pytest.approx(math.sqrt(2) / 2)


def test_half_extent():
    assert half_extent(0.0) == pytest.approx(0.5)
    assert half_extent(math.pi / 4) == pytest.approx(math.sqrt(2) / 2)


def test_sep_disjoint_axis_aligned():
    # Two unit squares centers 1.0 apart horizontally, touching on edge
    s = sep_pair(0.5, 0.5, 0.0, 1.5, 0.5, 0.0)
    assert s == pytest.approx(0.0, abs=1e-15)


def test_sep_overlapping():
    s = sep_pair(0.5, 0.5, 0.0, 1.0, 0.5, 0.0)
    assert s < 0.0
    assert s == pytest.approx(-0.5, abs=1e-12)


def test_sep_separated():
    s = sep_pair(0.5, 0.5, 0.0, 2.0, 0.5, 0.0)
    assert s == pytest.approx(0.5, abs=1e-12)


def test_bounding_box_single():
    squares = np.array([[0.5, 0.5, 0.0]], dtype=np.float64)
    xmin, xmax, ymin, ymax = bounding_box(squares)
    assert (xmin, xmax, ymin, ymax) == pytest.approx((0.0, 1.0, 0.0, 1.0))
    assert container_length_float64(squares) == pytest.approx(1.0)
