"""Search-program genome mutated by the Phase 5 evolution loop (section 9).

Mutations touch hyperparameters of the section-7 engine (move probabilities,
shrink schedule, temperature, init slack) — not raw packings.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Keep names local so importing this module under a tight sandbox RLIMIT_AS
# does not pull scipy via squarepack.search.anneal.
MOVE_NAMES = (
    "translate",
    "rotate",
    "jiggle",
    "swap_orientation",
    "relocate",
    "shift_row",
)

DEFAULT_MOVE_PROBS = (0.25, 0.2, 0.2, 0.1, 0.15, 0.1)


@dataclass
class SearchProgram:
    """Serializable search program (section 7 knobs)."""

    name: str = "section7_default"
    eta0: float = 1e-2
    temperature: float = 1e-3
    cool_factor: float = 0.995
    seeded_expand: float = 1.002
    max_lbfgs_iter: int = 80
    energy_feasible: float = 1e-24
    move_probs: list[float] = field(
        default_factory=lambda: list(DEFAULT_MOVE_PROBS)
    )
    # Bookkeeping
    parent: str | None = None
    method: str = "seed"  # seed | stub_mutate | llm_mutate
    generation: int = 0
    notes: str = ""

    def normalized_move_probs(self) -> list[float]:
        p = [max(1e-6, float(x)) for x in self.move_probs]
        if len(p) != len(MOVE_NAMES):
            # pad / trim to MOVE_NAMES
            p = (p + list(DEFAULT_MOVE_PROBS))[: len(MOVE_NAMES)]
        s = sum(p)
        return [x / s for x in p]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["move_probs"] = self.normalized_move_probs()
        d["move_names"] = list(MOVE_NAMES)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchProgram":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        if "move_probs" in payload:
            payload["move_probs"] = list(payload["move_probs"])
        return cls(**payload)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "SearchProgram":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def section7_seed_program() -> SearchProgram:
    """Baseline genome: the section-7 engine defaults."""
    return SearchProgram(name="section7_default", method="seed", generation=0)
