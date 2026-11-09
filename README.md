# squarepack

Pack `n` unit squares into the smallest axis-aligned square for `n = 1..1000`,
with every stored result fully verified.

## Phase 1 status

Geometry, verifier, atomic best-result store, and axis-aligned grid baseline.

```bash
pip install -e ".[dev]"
pytest
python scripts/build_grid_baseline.py --n-max 1000
```

Verified packings live under `results/best/nNNNN.json` and are written only
after full verification succeeds.
