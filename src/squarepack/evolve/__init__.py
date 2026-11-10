"""Program-evolution loop (Phase 5, section 9).

Heavy imports (fitness / anneal / scipy) are lazy so sandbox helpers can load
without mapping large native libraries under tight RLIMIT_AS.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EvolveResult",
    "SearchProgram",
    "detect_llm_provider",
    "evolve",
    "make_mutator",
    "section7_seed_program",
    "split_ns",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - import redirect
    if name in {"EvolveResult", "evolve", "split_ns"}:
        from squarepack.evolve import loop as _loop

        return getattr(_loop, name)
    if name in {"detect_llm_provider", "make_mutator"}:
        from squarepack.evolve import mutate as _mutate

        return getattr(_mutate, name)
    if name in {"SearchProgram", "section7_seed_program"}:
        from squarepack.evolve import program as _program

        return getattr(_program, name)
    raise AttributeError(name)
