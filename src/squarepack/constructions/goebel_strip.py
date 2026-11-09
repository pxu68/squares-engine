"""Goebel-type 45-degree diagonal strip with axis-aligned triangular fills."""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np


def goebel_strip_packing(
    strip_len: int,
    lower_rows: int,
    upper_rows: int,
    *,
    offset: float = 0.0,
) -> np.ndarray | None:
    """Build one Goebel-style packing.

    Parameters
    ----------
    strip_len :
        Number of 45-degree unit squares along the diagonal strip.
    lower_rows, upper_rows :
        Axis-aligned staircase / triangle fill depths on each side of the strip.
    offset :
        Extra translation of the strip along its axis (in unit-square steps).

    Geometry (assumption documented in ASSUMPTIONS.md):
    - Container side grows with the strip and staircases.
    - Strip squares are centered along the line y = x with theta = pi/4.
    - Lower triangle: axis-aligned squares with row r having (r+1) squares
      for r = 0..lower_rows-1 under the strip.
    - Upper triangle: mirrored above the strip.
    """
    if strip_len < 1 or lower_rows < 0 or upper_rows < 0:
        return None

    squares: list[list[float]] = []
    # Diagonal strip: consecutive 45° squares. Adjacent diamond centers are
    # separated by 1 along the strip axis (direction (1,1)/sqrt(2) projected
    # as spacing sqrt(2) in x=y coordinates between centers of touching diamonds
    # that share a flat? For unit squares at 45°, the projected width on x is
    # sqrt(2). Centers of a strip of diamonds touching edge-to-edge along the
    # diagonal are spaced 1 apart in both x and y when they share a side that
    # is perpendicular to the diagonal... Actually two unit squares at pi/4
    # whose centers differ by (1, 0) in strip-local coords after rotation:
    # Local strip: place squares with centers (0.5 + i, 0.5) then rotate by 45
    # about origin and translate.
    #
    # Simpler constructive form used here (common Goebel picture for n=5 etc.):
    # strip of `strip_len` diamonds along y=x, centers at
    #   ((i + 0.5)*sqrt(2)/2 + c, (i + 0.5)*sqrt(2)/2 + c) wait.
    #
    # Use: centers at (a + i, a + i) with spacing chosen so diamonds touch.
    # Support radius along diagonal direction phi=pi/4 equals 0.5 for a square
    # at theta=pi/4? r(pi/4, pi/4) = 0.5*(|1|+|0|)=0.5. Distance between
    # centers along the diagonal needed for touching: 1.0. So centers differ by
    # (1/sqrt(2), 1/sqrt(2)).
    step = 1.0 / math.sqrt(2.0)
    base = 0.5 * math.sqrt(2.0) + float(offset) * step
    for i in range(strip_len):
        c = base + i * step
        squares.append([c, c, math.pi / 4])

    # Axis-aligned staircase below the strip (lower-left).
    # Place rows of length 1,2,...,lower_rows with bottom-left at origin.
    for r in range(lower_rows):
        for c in range(r + 1):
            squares.append([0.5 + c, 0.5 + r, 0.0])

    # Axis-aligned staircase above/right of the strip.
    # Shift into the upper-right by roughly strip extent.
    # Bounding extent of strip along x or y:
    strip_max = base + (strip_len - 1) * step + 0.5 * math.sqrt(2.0)
    origin_u = strip_max - lower_rows  # heuristic alignment; verifier decides
    # Better: place upper staircase with its far corner at (S,S) where S is
    # chosen as ceil of needed size after assembly — instead place relative to
    # strip_max and let normalization/L computation handle the container.
    for r in range(upper_rows):
        for c in range(r + 1):
            # Mirror into top-right: growing toward increasing x,y from a corner
            # near the strip's far end.
            squares.append(
                [
                    strip_max - 0.5 - c,
                    strip_max - 0.5 - r,
                    0.0,
                ]
            )

    if not squares:
        return None
    return np.asarray(squares, dtype=np.float64)


def iter_goebel_params(
    *,
    max_strip: int = 12,
    max_rows: int = 8,
) -> Iterator[tuple[int, int, int, float]]:
    for strip in range(1, max_strip + 1):
        for lo in range(0, max_rows + 1):
            for hi in range(0, max_rows + 1):
                yield strip, lo, hi, 0.0


def goebel_candidates_for_n(
    n: int,
    *,
    max_strip: int = 12,
    max_rows: int = 8,
    limit: int = 40,
) -> list[np.ndarray]:
    """Generate up to `limit` Goebel packings with at least n squares, trimmed to n."""
    from squarepack.constructions.removal import remove_to_n

    out: list[np.ndarray] = []
    for strip, lo, hi, off in iter_goebel_params(max_strip=max_strip, max_rows=max_rows):
        pack = goebel_strip_packing(strip, lo, hi, offset=off)
        if pack is None or pack.shape[0] < n:
            continue
        trimmed = remove_to_n(pack, n)
        if trimmed is not None:
            out.append(trimmed)
            if len(out) >= limit:
                break
    return out
