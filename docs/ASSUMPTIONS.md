# Assumptions

Assumptions made while implementing `SQUARE_PACKING_SPEC.md`. Ambiguities are recorded here rather than chosen silently (agent rule 7).

## L definition and storage

1. **Container length L** follows section 2: `L = max(xmax - xmin, ymax - ymin)` over all corners of all unit squares.
2. **Verified L** (section 4.4) is computed with `mpmath` at 50 decimal digits, treating stored float64 `(x, y, theta)` as exact inputs. The high-precision value is then converted to float64 with **round-toward-+∞**: if the nearest float64 is already ≥ the high-precision value it is kept; otherwise `numpy.nextafter` toward `+inf` is used. The stored L is therefore never below the high-precision geometric side length.
3. **Containment** in full verification is checked against this stored (rounded-up) L, not against the raw high-precision value.
4. Packings are compared for promotion **only** by their stored verified L; a candidate replaces the stored best iff its verified L is **strictly smaller**.

## Normalization

5. Default normalization is **on** (`verification.normalize: true` in `config.yaml`): before dilation and L computation, the packing is translated so the axis-aligned bounding box has `xmin = ymin = 0`. Angles are unchanged.
6. After a nonzero dilation `c_i ← (1 + delta) * c_i`, the packing is re-normalized the same way so the stored container remains `[0, L]²`.

## Dilation

7. Full verification tries deltas in order: `0`, then `delta_default` (1e-12), then geometric growth by `delta_growth` (10×) up to `delta_max`. The **smallest** delta that passes both float64 (no tolerance) and mpmath checks is recorded.
8. Grid packings are expected to pass at `delta = 0` because they are exactly axis-aligned with float-exact coordinates.
9. **Imports** raise `delta_max` to `1e-4` (config `import.delta_max`) because published SVG coordinates can require larger dilation than constructed grids.

## Fast filter vs full verification

10. The fast filter uses a cell list with default cell size `1.5` and tolerances `sep >= -1e-9`, containment within `1e-9`. It never writes to `results/best/`. It may be given an external container side `L` (used by search later).
11. Full verification checks **all** pairs in float64 with **no** tolerance (`sep >= 0`, margins `>= 0`), then re-checks pairs with float64 `sep <= 1e-6` (and all containment margins) at 50 digits.
12. Because full verification **recomputes** L from corners (section 4.4) after normalization, containment against that self-derived L is satisfied whenever the geometry is consistent (margins ≥ 0, typically 0 on the bounding edges). Overlap is the primary negative-control failure mode for promotion.

## Grid baseline

13. Grid construction uses `k = ceil(sqrt(n))`, places the first `n` cells of a row-major `k × k` lattice with centers `(0.5 + i % k, 0.5 + i // k)` and `theta = 0`.
14. Grid results are stored with `origin: constructed`, `method: grid`, `seed: null`, empty attribution `source` when they are the best.

## Store format

15. Best results live at `results/best/nNNNN.json` matching section 10.2. History keeps the last 5 **previous** best records per `n` in `results/best/nNNNN.history.json` as `{"n": N, "history": [...]}`.
16. Writes are atomic: temp file in the same directory, `fsync`, `os.replace`.
17. `store.promote` returns the verified record payload even when not promoted (equal/worse L), so callers can archive imported seeds under `results/candidates/imported/` without clobbering a better best.

## Owner-confirmed (2026-11-09)

18. **L / container / CSV are placeholders.** Treat `L = max(bbox width, height)`, container `[0, L]²`, and the section-10.3 CSV layout as working defaults only. All submission formatting must be isolated in `export.py` (Phase 4). **Do not claim a submission is grader-ready** until one exported file has been checked against the official grader, sample, or validator.
19. **Imported published records are seeds only — never submit them as-is.** Store imported packings with `origin: imported`. `export` must support excluding `imported` results.

## Phase 2 — record import

20. **Primary source:** David Ellsworth’s “Squares in Squares” index  
    `https://kingbird.myphotos.cc/packing/squares_in_squares.html` (continuation of Erich Friedman’s Packing Center). `robots.txt` disallows unrelated CMS paths; `/packing/` is allowed. Fetches use a descriptive User-Agent and **≥ 1 request/second**.
21. **SVG parser:** Vendored `parse_svg_packing.py` from  
    `https://github.com/Davidebyzero/packing_squares_in_squares__tools` (no LICENSE file published upstream at import time) as `src/squarepack/records/ellsworth_parse_svg.py`, used only to parse that author’s SVGs. Coordinate conversion: SVG y-down → y-up via `Y = s - y` (non-centered) or centered flip; angles mapped as in the upstream CLI (`θ ← π/2 − θ`, then folded into `[0, π/2)`).
22. **Published L** is taken from the index-page `$s = ...$` field when a decimal/`\Nn{...}` form is present; otherwise from the SVG `<!ENTITY s "...">` value.
23. Index coverage is **sparse** in `1..324` (many perfect-square / near-square `n` omitted on the page). Only primary box SVGs are imported (one per displayed `n`). Alternate files such as `square-85b.svg` are not auto-imported unless they are the box link.
24. Large published `n` beyond 1000 (e.g. 1453, 1765, …) are ignored when `--n-max 1000`.
25. Parse count must equal `n`; mismatches go to `data/import_failures.csv` without guessing.
26. A packing that verifies but does **not** beat the stored best is still counted as verified in `data/records_snapshot.csv` and saved under `results/candidates/imported/` for seeding.

## Phase 2 — constructions

27. **Monotonic removal** deletes boundary-first squares (then farthest from centroid) from a larger packing; L is always recomputed by the verifier (bbox may shrink).
28. **L-extension** adds `2*floor(s)+1` axis-aligned unit squares along the +x and +y borders of a verified packing of side `s`, targeting container side `s+1`. When `s` is non-integral, the fractional strip is left empty (assumption).
29. **Goebel-type strip** is a parameterized constructive family (diagonal 45° strip + staircase fills), not a bit-exact reproduction of every historical Goebel packing. Candidates are verifier-driven; many parameter sets are rejected.
30. **Corner-rectangle compositions** place two opposite axis-aligned rectangles and fill the remaining cells of an `S×S` grid (`S = max(w1+w2, h1+h2)`). Optional tilt mode returns the two blocks only (no fill) as a seed.
31. **Neighbor seeds** in Phase 2 are geometric only (remove-one / insert-into-largest-hole). Section-7 local relaxation is **not** applied until Phase 3.
32. Expensive Goebel/composition scans default to `n ≤ construction_scan_cap` (150) in `scripts/build_baseline.py`; removal and L-extension run for all `n ≤ 1000`.

## Phase scope

33. Phase 3+ (anneal search, scheduler, polish, evolve, section 12) is not implemented yet. Package stubs exist for layout compatibility.
