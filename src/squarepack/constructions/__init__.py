"""Cheap constructive baselines (section 6)."""

from squarepack.constructions.composition import (
    composition_candidates_for_n,
    corner_composition,
)
from squarepack.constructions.goebel_strip import (
    goebel_candidates_for_n,
    goebel_strip_packing,
)
from squarepack.constructions.grid import grid_packing
from squarepack.constructions.l_extension import l_extension
from squarepack.constructions.neighbor_seed import seed_from_larger, seed_insert_one
from squarepack.constructions.removal import remove_to_n, removal_order

__all__ = [
    "grid_packing",
    "remove_to_n",
    "removal_order",
    "l_extension",
    "goebel_strip_packing",
    "goebel_candidates_for_n",
    "corner_composition",
    "composition_candidates_for_n",
    "seed_from_larger",
    "seed_insert_one",
]
