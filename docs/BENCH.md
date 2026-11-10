# Bench notes (Phase 3)

Hardware for these numbers: cloud agent VM (single shared CPU), not the target Dell R740.
Default config budgets remain 600s (`n <= 100`) and 1800s (`n > 100`) per job; acceptance overrides are shorter.

## Reproduction (n = 2..30)

Seeded low-temperature anneal from the current best packing (`seed_type=repro_best`), early-stop when verified L is within `1e-6` of the snapshot/best target.

| Metric | Value |
|---|---|
| Range | n = 2..30 |
| Hits | **29 / 29** |
| Tolerance | `1e-6` |
| Per-n budget (cap) | 15 s |
| Log | `results/logs/phase3_acceptance.json` |

Every n with an imported Ellsworth snapshot (and every n whose best is a construction/grid) reproduced within tolerance in at least one seed.

## Seeded hit rates (n in 30..60)

Same seeded reproduction protocol; 3 seeds per n; budget cap 30 s.

| n | Target L (best) | Hits | Hit rate | Mean seconds / seed | Core-hours / seed |
|---|---:|---:|---:|---:|---:|
| 31 | 6.0 | 3/3 | 1.0 | ~0.02 | ~6e-6 |
| 37 | 6.598619609 | 3/3 | 1.0 | ~0.03 | ~8e-6 |
| 41 | 6.926693094 | 3/3 | 1.0 | ~0.03 | ~8e-6 |
| 47 | 7.0 | 3/3 | 1.0 | ~0.04 | ~1e-5 |
| 53 | 7.822875656 | 3/3 | 1.0 | ~0.04 | ~1e-5 |

Seeded reproduction is essentially free once a verified best exists: the engine keeps the seed and confirms it under the anneal wrapper.

## Cold-start probe (structured / random, no best seed)

Budget 20 s × 2 seeds. These do **not** hit the record in 20 s; useful only as a lower bound on core-hours for unseeded search.

| n | Target | Hits | Mean seconds | Core-hours / seed | Best gap seen |
|---|---:|---:|---:|---:|---:|
| 31 | 6.0 | 0/2 | 20.1 | 0.0056 | +0.0064 |
| 37 | 6.5986 | 0/2 | 20.1 | 0.0056 | (no feasible) |
| 41 | 6.9267 | 0/2 | 20.0 | 0.0056 | +0.084 |

Log: `results/logs/phase3_cold_bench.json`.

**Implication for Phase 4:** plan on seeded low-temperature runs from imported/constructed baselines; cold starts need much larger budgets (config defaults 10–30 min) and many seeds. Rough R740 planning: at ~0.006 core-hours per 20 s probe, a 600 s job is ~0.17 core-hours per `(n, seed)` before accounting for promotion/verify.

## Scheduler smoke

File queue under `results/logs/queue/` with 2 workers, `--budget-s 5`, n∈{5,6,10,11}: 4 jobs completed (`status=done`). Basin files written under `results/logs/basin_n*.json`.

---

# Phase 4

Hardware: same cloud agent VM. Seeded low-temperature anneal + Newton polish sample (not a full R740 sweep).

## Seeded search sample

| Range | Eligible pool (skip near-squares) | Sampled | Budget / n | Promoted | Improved vs Phase-2 baseline | Mean seconds |
|---|---:|---:|---:|---:|---:|---:|
| 51..324 | 241 | 18 | 20 s | 0 | **0** | 23.5 |
| 325..1000 | 637 | 12 | 15 s | 0 | **0** | 44.0 |

- Mid sample n: 60, 71, 89, 109, 125, 145, 151, 164, 183, 202, 209, 232, 248, 263, 274, 290, 307, 312  
- High sample n: 367, 408, 438, 498, 600, 636, 688, 769, 815, 871, 891, 992  
- Log: `results/logs/phase4_search_summary.json`  
- Finding: short seeded runs did not beat the Phase-2 baseline; longer budgets / more seeds needed on the R740 for real improvement hunting.

## Export self-check

- `scripts/export_packings.py --all --exclude-origin imported` → 892 packings (imports excluded).
- Re-verify from `squares.csv` alone: **892 / 892 passed** (~194 s wall with 4 workers).
- Evidence: `results/export/no_imported/export_reverify.json`, `export_meta.json`, `L_by_n.csv` (large `squares.csv` gitignored).
