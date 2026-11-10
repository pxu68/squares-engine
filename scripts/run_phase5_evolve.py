#!/usr/bin/env python3
"""Phase 5: program-evolution loop (section 9).

If no LLM API key is present, runs a stub-mutator dry-run that still exercises
fitness, train/held-out split, sandbox validation, and archival.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from squarepack.evolve.mutate import detect_llm_provider  # noqa: E402
from squarepack.evolve.loop import evolve  # noqa: E402
from squarepack.evolve.sandbox import run_python_sandboxed, validate_program_dict  # noqa: E402
from squarepack.evolve.program import section7_seed_program  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=_REPO)
    p.add_argument(
        "--ns",
        type=str,
        default="11,17,18,19,26,27,28,29,37,41,47,53",
        help="Comma list of n for train/held-out split",
    )
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--population", type=int, default=2)
    p.add_argument("--budget-s", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--force-stub", action="store_true", help="Ignore LLM keys if present")
    args = p.parse_args(argv)

    repo = args.repo_root.resolve()
    provider, var = detect_llm_provider()
    print(f"LLM provider detected: {provider} ({var})" if provider else "LLM provider detected: none")

    # Sandbox smoke: validate seed program JSON inside a limited subprocess.
    # Import only sandbox/program (no scipy) under a modest address-space cap.
    seed = section7_seed_program()
    sbx = run_python_sandboxed(
        "import json\n"
        f"data = json.loads({json.dumps(json.dumps(seed.to_dict()))})\n"
        "from squarepack.evolve.sandbox import validate_program_dict\n"
        "validate_program_dict(data)\n"
        "print(json.dumps({'ok': True, 'name': data.get('name')}))\n",
        cwd=repo,
        timeout_s=20,
        cpu_s=15,
        mem_mb=1024,
    )
    print("sandbox smoke:", sbx)
    validate_program_dict(seed.to_dict())

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    result = evolve(
        repo,
        ns=ns,
        generations=args.generations,
        population=args.population,
        time_budget_s=args.budget_s,
        seed=args.seed,
        prefer_llm=not args.force_stub,
    )
    out = repo / "results" / "logs" / "phase5_evolve_summary.json"
    out.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    print(
        json.dumps(
            {
                "mutator": result.mutator,
                "baseline_train": result.baseline_train,
                "baseline_heldout": result.baseline_heldout,
                "best_train": result.best_train,
                "best_heldout": result.best_heldout,
                "improved_heldout": result.improved_heldout,
                "notes": result.notes,
            },
            indent=2,
        )
    )
    # Exit 0 either way — acceptance allows documenting that the loop did not help.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
