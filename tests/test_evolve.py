"""Phase 5 evolve-loop unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from squarepack.evolve.fitness import score_trial
from squarepack.evolve.mutate import StubMutator, detect_llm_provider, make_mutator
from squarepack.evolve.program import SearchProgram, section7_seed_program
from squarepack.evolve.sandbox import SandboxError, run_python_sandboxed, validate_program_dict
from squarepack.evolve.loop import split_ns


def test_score_trial_penalizes_unverified_and_trivial_grid():
    s, _ = score_trial(L_found=None, L_best_known=4.0, grid_L=5.0, verified=False)
    assert s == 1.0
    s2, reason = score_trial(L_found=5.0, L_best_known=4.7, grid_L=5.0, verified=True)
    assert s2 == 1.0 and "trivial_grid" in reason
    s3, _ = score_trial(L_found=4.5, L_best_known=4.7, grid_L=5.0, verified=True)
    assert s3 < 0  # beat best known


def test_stub_mutator_changes_program():
    parent = section7_seed_program()
    child = StubMutator().mutate(parent, generation=1, context={}, rng=np.random.default_rng(0))
    assert child.name != parent.name
    assert child.method == "stub_mutate"
    assert child.generation == 1
    validate_program_dict(child.to_dict())


def test_validate_program_rejects_unknown_keys():
    with pytest.raises(SandboxError):
        validate_program_dict({"eta0": 0.1, "os_system": "rm -rf /"})


def test_sandbox_strips_and_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Build a tiny repo-like layout with src on path via cwd=/workspace in real runs;
    # here just execute a pure JSON print without importing squarepack.
    monkeypatch.setenv("OPENAI_API_KEY", "should-be-stripped")
    out = run_python_sandboxed(
        "import os, json\n"
        "print(json.dumps({'has_key': 'OPENAI_API_KEY' in os.environ}))\n",
        cwd=Path("/workspace"),
        timeout_s=10,
        cpu_s=5,
        mem_mb=128,
    )
    assert out["has_key"] is False


def test_split_ns_disjoint():
    train, held = split_ns(range(10, 20), heldout_frac=0.3, seed=1)
    assert set(train).isdisjoint(held)
    assert set(train) | set(held) == set(range(10, 20))


def test_make_mutator_defaults_to_stub_without_keys(monkeypatch: pytest.MonkeyPatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert detect_llm_provider() == (None, None)
    mut, name = make_mutator(prefer_llm=True)
    assert "stub" in name
    assert isinstance(mut, StubMutator)


def test_program_roundtrip(tmp_path: Path):
    p = section7_seed_program()
    path = tmp_path / "prog.json"
    p.save(path)
    q = SearchProgram.load(path)
    assert q.eta0 == p.eta0
    assert abs(sum(q.normalized_move_probs()) - 1.0) < 1e-12
