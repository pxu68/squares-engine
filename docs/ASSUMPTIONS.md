# Assumptions (Phase 1)

Assumptions made while implementing Phase 1 of `SQUARE_PACKING_SPEC.md`. Ambiguities are recorded here rather than chosen silently (agent rule 7).

## L definition and storage

1. **Container length L** follows section 2: `L = max(xmax - xmin, ymax - ymin)` over all corners of all unit squares.
2. **Verified L** (section 4.4) is computed with `mpmath` at 50 decimal digits, treating stored float64 `(x, y, theta)` as exact inputs. The high-precision value is then converted to float64 with **round-toward-+∞**: if the nearest float64 is already ≥ the high-precision value it is kept; otherwise `numpy.nextafter` toward `+inf` is used. The stored L is therefore never below the high-precision geometric side length.
3. **Containment** in full verification is checked against this stored (rounded-up) L, not against the raw high-precision value.
4. Packings are compared for promotion **only** by their stored verified L; a candidate replaces the stored best iff its verified L is **strictly smaller**.

## Normalization

5. Default normalization is **on** (`verification.normalize: true` in `config.yaml`): before dilation and L computation, the packing is translated so the axis-aligned bounding box has `xmin = ymin = 0`. Angles are unchanged.
6. After a nonzero dilation `c_i ← (1 + delta) * c_i`, the packing is re-normalized the same way so the stored container remains `[0, L]²`.

## Dilation

7. Full verification tries deltas in order: `0`, then `delta_default` (1e-12), then geometric growth by `delta_growth` (10×) up to `delta_max` (1e-6). The **smallest** delta that passes both float64 (no tolerance) and mpmath checks is recorded.
8. Grid packings are expected to pass at `delta = 0` because they are exactly axis-aligned with float-exact coordinates.

## Fast filter vs full verification

9. The fast filter uses a cell list with default cell size `1.5` and tolerances `sep >= -1e-9`, containment within `1e-9`. It never writes to `results/best/`. It may be given an external container side `L` (used by search later).
10. Full verification checks **all** pairs in float64 with **no** tolerance (`sep >= 0`, margins `>= 0`), then re-checks pairs with float64 `sep <= 1e-6` (and all containment margins) at 50 digits.
11. Because full verification **recomputes** L from corners (section 4.4) after normalization, containment against that self-derived L is satisfied whenever the geometry is consistent (margins ≥ 0, typically 0 on the bounding edges). Overlap is the primary negative-control failure mode for promotion.

## Grid baseline

12. Grid construction uses `k = ceil(sqrt(n))`, places the first `n` cells of a row-major `k × k` lattice with centers `(0.5 + i % k, 0.5 + i // k)` and `theta = 0`.
13. Phase 1 grid results are stored with `origin: constructed`, `method: grid`, `seed: null`, empty attribution `source`.

## Store format

14. Best results live at `results/best/nNNNN.json` matching section 10.2. History keeps the last 5 **previous** best records per `n` in `results/best/nNNNN.history.json` as `{"n": N, "history": [...]}`.
15. Writes are atomic: temp file in the same directory, `fsync`, `os.replace`.

## Owner-confirmed (2026-11-09)

16. **L / container / CSV are placeholders.** Treat `L = max(bbox width, height)`, container `[0, L]²`, and the section-10.3 CSV layout as working defaults only. All submission formatting must be isolated in `export.py` (Phase 4). **Do not claim a submission is grader-ready** until one exported file has been checked against the official grader, sample, or validator.
17. **Imported published records are seeds only — never submit them as-is.** Store imported packings with `origin: imported`. `export` must support excluding `imported` results. Phase 1 is unaffected (grid baselines use `origin: constructed`).

## Phase 1 scope

18. Record import, other constructions, anneal search, polish, scheduler, evolve, and section 12 are **not** implemented in Phase 1. Package stubs exist only to match the repo layout.
