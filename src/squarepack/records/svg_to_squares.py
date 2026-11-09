"""Convert Ellsworth SVG packings to (x, y, theta) arrays in y-up [0, s]² coords."""

from __future__ import annotations

from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

import numpy as np

from squarepack.records.ellsworth_parse_svg import hp_pi, parse_svg


def svg_to_packing(
    svg_path: Path | str,
    *,
    precision: int = 50,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Parse an Ellsworth SVG into float64 squares and published side length.

    Returns
    -------
    squares : (n, 3) array of (x, y, theta) in radians, y-up, roughly in [0, s]²
    L_published : float from the SVG ``s`` entity
    meta : entity dict and parse flags
    """
    svg_path = Path(svg_path)
    squares_raw, entities, actual_precision, viewbox_centered = parse_svg(
        str(svg_path), precision
    )
    if "s" not in entities:
        raise ValueError(f"SVG {svg_path} has no entity s (published side length)")

    getcontext().prec = actual_precision + 20
    pi_hp = hp_pi(actual_precision)
    threshold = (
        Decimal(10) ** (-(precision - 10)) if precision > 10 else Decimal("1e-6")
    )
    s_value = Decimal(entities["s"])
    half_s = s_value / Decimal(2)

    out: list[list[float]] = []
    for sq in squares_raw:
        th = pi_hp / Decimal(2) - sq["theta"]
        if abs(th) < threshold:
            th = Decimal(0)
        # Normalize angle into [0, pi/2) as the upstream CLI does.
        th = (th + pi_hp) % (pi_hp / Decimal(2))

        if viewbox_centered:
            # Centered SVG coords -> shift into [0, s], flip y to y-up.
            x = float(sq["x"] + half_s)
            y = float(-sq["y"] + half_s)
        else:
            # Non-centered SVG is already in [0, s] with y-down.
            x = float(sq["x"])
            y = float(s_value - sq["y"])
        out.append([x, y, float(th)])

    arr = np.asarray(out, dtype=np.float64)
    meta = {
        "entities": {k: str(v) for k, v in entities.items()},
        "actual_precision": int(actual_precision),
        "viewbox_centered": bool(viewbox_centered),
        "n_parsed": int(arr.shape[0]),
    }
    return arr, float(s_value), meta
