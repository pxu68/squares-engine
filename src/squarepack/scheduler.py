"""File-based search scheduler (section 8)."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from squarepack.search.anneal import anneal
from squarepack.search.basin import load_basin, record_run, should_deprioritize
from squarepack.store import BestStore, Candidate, load_config


def _is_square(n: int) -> bool:
    k = int(math.isqrt(n))
    return k * k == n


def default_skip_n(n: int, *, skip_near_square: bool = True) -> bool:
    """Skip perfect squares; optionally skip k^2-1 and k^2-2."""
    if _is_square(n):
        return True
    if not skip_near_square:
        return False
    k = int(math.ceil(math.sqrt(n)))
    return n in (k * k - 1, k * k - 2)


def time_budget_for_n(n: int, config: dict[str, Any] | None = None) -> float:
    cfg = (config or {}).get("search", {})
    if n <= 100:
        return float(cfg.get("budget_s_small", 600))
    return float(cfg.get("budget_s_large", 1800))


@dataclass(frozen=True)
class Job:
    n: int
    seed: int
    seed_type: str = "best"


def queue_dir(repo: Path) -> Path:
    return repo / "results" / "logs" / "queue"


def _job_path(qdir: Path, job: Job) -> Path:
    return qdir / f"job_n{job.n:04d}_seed{job.seed}_{job.seed_type}.json"


def enqueue(repo: Path, jobs: Iterable[Job]) -> int:
    qdir = queue_dir(repo)
    qdir.mkdir(parents=True, exist_ok=True)
    done = qdir / "done"
    done.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for job in jobs:
        path = _job_path(qdir, job)
        if path.exists() or (done / path.name).exists():
            continue
        payload = {"n": job.n, "seed": job.seed, "seed_type": job.seed_type, "status": "queued"}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        tmp.replace(path)
        n_new += 1
    return n_new


def build_round_robin_jobs(
    n_min: int,
    n_max: int,
    *,
    seeds_per_n: int = 1,
    seed0: int = 1,
    skip_near_square: bool = True,
    logs_dir: Path | None = None,
    max_runs_deprioritize: int = 200,
) -> list[Job]:
    """Round-robin over n still worth searching."""
    ns = [
        n
        for n in range(n_min, n_max + 1)
        if not default_skip_n(n, skip_near_square=skip_near_square)
    ]
    jobs: list[Job] = []
    for k in range(seeds_per_n):
        for n in ns:
            if logs_dir is not None:
                data = load_basin(logs_dir, n)
                if should_deprioritize(data, seed_type="best", max_runs=max_runs_deprioritize):
                    continue
            jobs.append(Job(n=n, seed=seed0 + 1000 * n + k, seed_type="best"))
    return jobs


def _claim_job(qdir: Path) -> dict[str, Any] | None:
    """Atomically claim one queued job via rename into running/."""
    running = qdir / "running"
    running.mkdir(parents=True, exist_ok=True)
    for path in sorted(qdir.glob("job_*.json")):
        dest = running / path.name
        try:
            os.rename(path, dest)
        except FileNotFoundError:
            continue
        except OSError:
            continue
        try:
            return json.loads(dest.read_text(encoding="utf-8")) | {"_path": str(dest)}
        except Exception:
            continue
    return None


def _finish_job(qdir: Path, job_path: Path, result: dict[str, Any]) -> None:
    done = qdir / "done"
    done.mkdir(parents=True, exist_ok=True)
    out = done / job_path.name
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    try:
        job_path.unlink(missing_ok=True)
    except OSError:
        pass


def _load_seed_squares(store: BestStore, n: int, seed_type: str) -> np.ndarray | None:
    if seed_type == "best":
        rec = store.load(n)
        if rec is None:
            return None
        return np.asarray(rec["squares"], dtype=np.float64)
    if seed_type == "neighbor_plus" and store.load(n + 1) is not None:
        from squarepack.constructions.neighbor_seed import seed_from_larger

        return seed_from_larger(np.asarray(store.load(n + 1)["squares"], dtype=np.float64), n)
    if seed_type == "neighbor_minus" and n > 1 and store.load(n - 1) is not None:
        from squarepack.constructions.neighbor_seed import seed_insert_one

        rec = store.load(n - 1)
        return seed_insert_one(np.asarray(rec["squares"], dtype=np.float64), float(rec["L"]))
    return None


def run_one_job(
    repo: Path,
    n: int,
    seed: int,
    seed_type: str = "best",
    *,
    time_budget_s: float | None = None,
    promote: bool = True,
) -> dict[str, Any]:
    config = load_config(repo / "config.yaml")
    store = BestStore(config=config)
    logs_dir = repo / config.get("paths", {}).get("results_logs", "results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    init = _load_seed_squares(store, n, seed_type)
    best_L = store.current_L(n)
    budget = time_budget_s if time_budget_s is not None else time_budget_for_n(n, config)

    result = anneal(
        n,
        seed,
        init_squares=init,
        best_known_L=best_L,
        time_budget_s=budget,
        temperature=float(config.get("search", {}).get("temperature", 1e-3)),
    )

    improved = False
    verified_L = None
    promoted = False
    if result.ok_fast and promote:
        ok, report, _ = store.promote(
            Candidate(
                n=n,
                squares=result.squares,
                origin="searched",
                method="anneal",
                seed=seed,
            )
        )
        promoted = bool(ok)
        if report.ok and report.L is not None:
            verified_L = float(report.L)
            improved = promoted
    elif result.ok_fast:
        # Fast-only path for reproduction tests
        verified_L = float(result.L)

    record_run(
        logs_dir,
        n=n,
        seed=seed,
        seed_type=seed_type,
        L_verified=verified_L,
        improved=improved,
        elapsed_s=result.elapsed_s,
        extras={"steps": result.steps, "s_final": result.s_final, "promoted": promoted},
    )
    return {
        "n": n,
        "seed": seed,
        "seed_type": seed_type,
        "L": result.L,
        "verified_L": verified_L,
        "improved": improved,
        "promoted": promoted,
        "elapsed_s": result.elapsed_s,
        "steps": result.steps,
        "ok_fast": result.ok_fast,
    }


def _worker_loop(repo_str: str, time_budget_override: float | None, idle_sleep: float) -> None:
    repo = Path(repo_str)
    qdir = queue_dir(repo)
    while True:
        job = _claim_job(qdir)
        if job is None:
            time.sleep(idle_sleep)
            # Exit if nothing queued/running
            if not any(qdir.glob("job_*.json")) and not any((qdir / "running").glob("job_*.json")):
                return
            continue
        path = Path(job["_path"])
        try:
            summary = run_one_job(
                repo,
                int(job["n"]),
                int(job["seed"]),
                str(job.get("seed_type", "best")),
                time_budget_s=time_budget_override,
                promote=True,
            )
            summary["status"] = "done"
        except Exception as exc:  # noqa: BLE001 — worker must not die silently
            summary = {
                "n": job.get("n"),
                "seed": job.get("seed"),
                "status": "error",
                "error": repr(exc),
            }
        _finish_job(qdir, path, summary)


def run_scheduler(
    repo: Path,
    *,
    workers: int = 1,
    time_budget_s: float | None = None,
    idle_sleep: float = 0.5,
) -> None:
    """Pull jobs from the file queue with ``workers`` processes."""
    workers = max(1, int(workers))
    procs: list[Process] = []
    for _ in range(workers):
        p = Process(target=_worker_loop, args=(str(repo), time_budget_s, idle_sleep))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
