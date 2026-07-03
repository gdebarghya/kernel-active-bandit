cat > CLAUDE.md << 'EOF'
# kernel-active-bandit

Benchmark suite comparing KABC (our nonparametric active clustering method) against
ACB, TaS-FW, and baselines, on synthetic Gaussian mixtures and MovieLens-1M.

Full experimental design: see `docs/experiments.md` — read it before implementing
any algorithm or experiment logic.

## Environment

- Conda env `kabc`, Python 3.13. Activate before running anything: `conda activate kabc`
- Install project in editable mode: `pip install -e ".[dev]"`
- Jupyter kernel is registered as "Python (kabc)" — use it, not base

## Algorithm interface (src/kabc/algorithms/)

Every algorithm implements:
`fit(env, delta, budget_cap, rng) -> Result(partition, tau, stopped, history)`
Environments implement `pull(arm_index, rng) -> sample`.
New algorithms must conform to this interface — do not special-case one algorithm's
calling convention in experiment scripts.

## Non-negotiable protocol rules

- Pilot pulls (median-heuristic bandwidth estimation, n_pilot=5/arm) COUNT toward
  the reported budget tau. Never exclude them.
- Calibrated threshold constants (c_cal) are frozen once in `configs/calibration.yaml`
  by `scripts/calibrate_constants.py`. Do not re-tune per experiment — reuse the
  frozen value across Exp 1/2/3, per the calibration protocol in docs/experiments.md.
- Notebooks in `notebooks/` only load results from `results/*.parquet` and plot —
  they must never silently recompute. If results are missing, fail loudly and point
  to the `make expN` target that generates them.

## Commands

- `make data` — fetches/verifies MovieLens-1M in data/ml-1m/
- `make calibrate` — freezes c_cal into configs/calibration.yaml
- `make smoke` — all 3 experiments, reduced grid/reps, sanity check before a full run
- `make exp1` / `exp2` / `exp3` — full experiment runs
- `make test` — pytest tests/

## Repo conventions

- `data/` and `results/` are gitignored and claudeignored — never read result parquet
  files directly into context; summarize via pandas in a script/notebook cell instead
- Config (grids, seeds, budget caps) lives in `configs/*.yaml`, not hardcoded in `src/`
EOF