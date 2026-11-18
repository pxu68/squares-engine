# Run

Commands assume the **repository root** as the working directory (`cd` into the clone first). Python 3.11+.

## Install and test

```bash
pip install -e ".[dev]"
pytest
```

## Baselines

```bash
# Phase 1 — axis-aligned grid into results/best/
python scripts/build_grid_baseline.py --n-max 1000

# Phase 2 — imports + constructions → results/baseline.csv
python scripts/build_baseline.py --n-min 1 --n-max 1000
# or reuse cached imports:
python scripts/build_baseline.py --skip-import
```

## Search and scheduler

```bash
# File-queue scheduler (section 8)
python scripts/run_scheduler.py --n-min 2 --n-max 30 --seeds-per-n 1 --workers 1 --budget-s 60

# Phase 3 acceptance / bench
python scripts/run_phase3_acceptance.py --n-min 2 --n-max 30 --budget-s 15 \
  --bench-ns 31,37,41,47,53 --bench-budget-s 30
```

Useful flags: `--enqueue-only` / `--run-only` on the scheduler; `--skip-bench` / `--promote` on phase-3 acceptance. See `python scripts/<name>.py --help`.

## Phase 4 — seeded search, polish, export

```bash
# Stratified sample over 51..324 and 325..1000 (see docs/BENCH.md)
python scripts/run_phase4_search.py --sample-mid 18 --sample-high 12 \
  --budget-mid-s 20 --budget-high-s 15

# Export (imports are seeds only — exclude them for submission-shaped output)
python scripts/export_packings.py --all --exclude-origin imported \
  --out-dir results/export/no_imported
```

Partial export: `--n 1,5,17` or `--range 1 200`. Self-check re-verifies from the CSV by default (`--no-self-check` to skip).

## Phase 5 — evolve

Requires no coding changes if an LLM key is absent: use `--force-stub` for the dry-run harness.

```bash
python scripts/run_phase5_evolve.py --force-stub --generations 3 --population 2 --budget-s 5
```

Optional: `--ns 11,17,18,19,26,27,28,29,37,41,47,53`. With `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY` set, omit `--force-stub` to use the LLM mutator.

## Notes

- Tunables: `config.yaml`. Assumptions and measured budgets: `docs/ASSUMPTIONS.md`, `docs/BENCH.md`.
- Stored packings: `results/best/nNNNN.json` (written only after full verification).
