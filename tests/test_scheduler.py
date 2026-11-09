"""Scheduler helpers."""

from __future__ import annotations

from pathlib import Path

from squarepack.scheduler import Job, build_round_robin_jobs, default_skip_n, enqueue


def test_skip_perfect_and_near_squares():
    assert default_skip_n(9) is True
    assert default_skip_n(8) is True  # 9-1
    assert default_skip_n(7) is True  # 9-2
    assert default_skip_n(5) is False


def test_enqueue_round_robin(tmp_path: Path):
    # Minimal fake repo layout
    (tmp_path / "results" / "logs").mkdir(parents=True)
    jobs = build_round_robin_jobs(5, 6, seeds_per_n=1, seed0=1, skip_near_square=True)
    assert all(j.n in (5, 6) for j in jobs)
    n = enqueue(tmp_path, jobs)
    assert n == len(jobs)
    assert enqueue(tmp_path, jobs) == 0  # idempotent
