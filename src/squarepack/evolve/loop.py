"""Program-evolution loop (section 9)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from squarepack.evolve.fitness import FitnessReport, evaluate_program
from squarepack.evolve.mutate import Mutator, make_mutator
from squarepack.evolve.program import SearchProgram, section7_seed_program
from squarepack.store import BestStore


@dataclass
class GenerationRecord:
    generation: int
    program: dict[str, Any]
    train: dict[str, Any]
    heldout: dict[str, Any]
    accepted: bool
    reason: str


@dataclass
class EvolveResult:
    baseline_train: float
    baseline_heldout: float
    best_program: dict[str, Any] | None
    best_train: float | None
    best_heldout: float | None
    improved_heldout: bool
    mutator: str
    generations: list[GenerationRecord] = field(default_factory=list)
    archive_dir: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_train": self.baseline_train,
            "baseline_heldout": self.baseline_heldout,
            "best_program": self.best_program,
            "best_train": self.best_train,
            "best_heldout": self.best_heldout,
            "improved_heldout": self.improved_heldout,
            "mutator": self.mutator,
            "generations": [
                {
                    "generation": g.generation,
                    "program": g.program,
                    "train": g.train,
                    "heldout": g.heldout,
                    "accepted": g.accepted,
                    "reason": g.reason,
                }
                for g in self.generations
            ],
            "archive_dir": self.archive_dir,
            "notes": self.notes,
        }


def split_ns(
    ns: Sequence[int],
    *,
    heldout_frac: float = 0.35,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    arr = np.array(sorted(set(int(n) for n in ns)), dtype=int)
    rng.shuffle(arr)
    n_hold = max(1, int(math.ceil(len(arr) * heldout_frac)))
    held = sorted(arr[:n_hold].tolist())
    train = sorted(arr[n_hold:].tolist())
    if not train:
        # tiny sets: put one in train
        train, held = held[:1], held[1:] or held[:1]
    return train, held


def _load_baseline_L(repo: Path) -> dict[int, float]:
    import csv

    path = repo / "results" / "baseline.csv"
    out: dict[int, float] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["n"])] = float(row["L"])
    return out


def _basin_summary(logs_dir: Path, limit: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not logs_dir.is_dir():
        return out
    for path in sorted(logs_dir.glob("basin_n*.json"))[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "n": data.get("n"),
                "n_runs": len(data.get("runs") or []),
                "n_basins": len(data.get("histogram") or {}),
            }
        )
    return out


def evolve(
    repo: Path,
    *,
    ns: Sequence[int],
    generations: int = 3,
    population: int = 2,
    time_budget_s: float = 6.0,
    seed: int = 20261109,
    prefer_llm: bool = True,
    archive_dirname: str = "evolve_archive",
) -> EvolveResult:
    """Run the evolution loop; accept a child only if held-out score improves."""
    repo = Path(repo)
    store = BestStore(config={})
    # Reload with repo config
    from squarepack.store import load_config

    store = BestStore(config=load_config(repo / "config.yaml"))
    if not store.best_dir.is_absolute():
        store.best_dir = repo / store.best_dir

    baseline = _load_baseline_L(repo)
    train_ns, held_ns = split_ns(ns, seed=seed)
    logs_dir = repo / "results" / "logs"
    archive = repo / "results" / "logs" / archive_dirname
    archive.mkdir(parents=True, exist_ok=True)

    mutator, mutator_name = make_mutator(prefer_llm=prefer_llm)
    rng = np.random.default_rng(seed)

    seed_prog = section7_seed_program()
    seed_prog.save(archive / "section7_default.json")

    base_train = evaluate_program(
        seed_prog, train_ns, store, time_budget_s=time_budget_s, seed0=seed, baseline=baseline
    )
    base_hold = evaluate_program(
        seed_prog, held_ns, store, time_budget_s=time_budget_s, seed0=seed + 1, baseline=baseline
    )
    (archive / "baseline_train.json").write_text(
        json.dumps(base_train.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (archive / "baseline_heldout.json").write_text(
        json.dumps(base_hold.as_dict(), indent=2) + "\n", encoding="utf-8"
    )

    best_prog: SearchProgram | None = None
    best_train = base_train.mean_score
    best_held = base_hold.mean_score
    improved = False
    generations_out: list[GenerationRecord] = []
    parent = seed_prog

    context = {
        "basin_summary": _basin_summary(logs_dir),
        "failure_notes": [
            "Do not emit trivial grid packings; fitness assigns no credit unless L < grid.",
            "Mutate hyperparameters only.",
        ],
        "train_score": base_train.mean_score,
        "heldout_score": base_hold.mean_score,
    }

    for gen in range(1, generations + 1):
        children: list[SearchProgram] = []
        for _ in range(population):
            child = mutator.mutate(parent, generation=gen, context=context, rng=rng)
            child.save(archive / f"{child.name}.json")
            children.append(child)

        # Evaluate each child; keep the best held-out among those that also beat train baseline
        gen_best: SearchProgram | None = None
        gen_train: FitnessReport | None = None
        gen_hold: FitnessReport | None = None
        for child in children:
            tr = evaluate_program(
                child, train_ns, store, time_budget_s=time_budget_s, seed0=seed + gen, baseline=baseline
            )
            ho = evaluate_program(
                child,
                held_ns,
                store,
                time_budget_s=time_budget_s,
                seed0=seed + gen + 99,
                baseline=baseline,
            )
            (archive / f"{child.name}_train.json").write_text(
                json.dumps(tr.as_dict(), indent=2) + "\n", encoding="utf-8"
            )
            (archive / f"{child.name}_heldout.json").write_text(
                json.dumps(ho.as_dict(), indent=2) + "\n", encoding="utf-8"
            )
            # Accept only if BOTH train and held-out improve over section-7 baseline.
            ok = (tr.mean_score < base_train.mean_score - 1e-12) and (
                ho.mean_score < base_hold.mean_score - 1e-12
            )
            reason = "accepted" if ok else "heldout_or_train_not_improved"
            if gen_hold is None or ho.mean_score < gen_hold.mean_score:
                gen_best, gen_train, gen_hold = child, tr, ho
            generations_out.append(
                GenerationRecord(
                    generation=gen,
                    program=child.to_dict(),
                    train=tr.as_dict(),
                    heldout=ho.as_dict(),
                    accepted=ok,
                    reason=reason,
                )
            )
            if ok and (best_prog is None or ho.mean_score < float(best_held)):
                best_prog = child
                best_train = tr.mean_score
                best_held = ho.mean_score
                improved = True
                parent = child
                context["train_score"] = tr.mean_score
                context["heldout_score"] = ho.mean_score

        # Diversity archive: keep gen-best even if not accepted
        if gen_best is not None:
            gen_best.save(archive / f"gen{gen}_best.json")

    notes = (
        f"mutator={mutator_name}; train_ns={train_ns}; held_ns={held_ns}; "
        f"budget_s={time_budget_s}; improved_heldout={improved}"
    )
    result = EvolveResult(
        baseline_train=base_train.mean_score,
        baseline_heldout=base_hold.mean_score,
        best_program=None if best_prog is None else best_prog.to_dict(),
        best_train=None if best_prog is None else best_train,
        best_heldout=None if best_prog is None else best_held,
        improved_heldout=improved,
        mutator=mutator_name,
        generations=generations_out,
        archive_dir=str(archive),
        notes=notes,
    )
    (archive / "evolve_summary.json").write_text(
        json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return result
