# kernel-active-bandit

Benchmark suite for **KABC** (Kernel-based Active Clustering of Bandit Arms) —
a nonparametric method for actively clustering bandit arms based on their full
distributions, not just means. Companion code for our paper (arXiv 2601.07535).

Compares KABC against parametric baselines (ACB, TaS-FW) and a uniform-sampling
baseline, on synthetic Gaussian mixtures and real MovieLens-1M rating data,
including a dedicated experiment (Experiment 4) testing whether KABC's
decision rule holds up better than kernel k-means under real, resampled
cluster-size imbalance.

| Experiment | Setting | Algorithms |
|---|---|---|
| 1 | Gaussian mixtures (sweeps over Δ, d, N) | kabc (+2x/0.5x calibration ablation, sg), acb, acb-gaussian |
| 2 | Same-mean, different-shape pairs | kabc (+2x/0.5x, sg), acb |
| 3 | MovieLens random movie pairs (N=8, K=2) | kabc, acb, tas-fw, unif-kkm |
| 4 | MovieLens prototypes with cluster-size imbalance (K=7, 12) | kabc, unif-kkm, acb |

Full experimental design, algorithm details, and figure specs: see
[`docs/experiments.md`](docs/experiments.md).

## Setup

```bash
conda create -n kabc python=3.13
conda activate kabc
pip install -e ".[dev]"
```

Register the Jupyter kernel:
```bash
python -m ipykernel install --user --name kabc --display-name "Python (kabc)"
```

MovieLens-1M: place the `ml-1m/` folder (from grouplens.org) under `data/`, or
run `make data` to auto-download it.

## Usage

```bash
make data           # fetch/verify MovieLens-1M
make movielens-207  # build data/ml-1m-207/ (the >=1000-rating movie pool experiments 3-4 sample from)
make calibrate       # freeze practical threshold constants
make smoke           # quick sanity run, experiments 1-2, reduced grid
make smoke-exp3      # separate, slower sanity run for experiment 3 (includes tas-fw)
make smoke-exp4      # separate, slower sanity run for experiment 4 (real k=8 pathway)
make exp1            # Experiment 1: Gaussian mixtures
make exp2            # Experiment 2: same-mean, different-shape instances
make exp3            # Experiment 3: MovieLens random pairs
make exp4            # Experiment 4: MovieLens prototypes, kabc vs. unif-kkm (and acb) under imbalance
make exp2-acb        # ACB-only rerun on Experiment 2's existing instances (replaces the acb rows)
make exp4-acb        # ACB-only pass on Experiment 4's existing instances (adds the acb rows)
make test            # run test suite
```

`make exp4` takes many hours (kabc's doubling rounds at k=8 dominate); the others take minutes
to roughly an hour on a laptop. `make exp2-acb` / `make exp4-acb` expect the corresponding main
run to exist already and back up the results file before merging.

Notebooks in `notebooks/` load results from `results/*/results.parquet` and produce
paper figures (written to `results/figures/`) — run the corresponding `make expN` first:

| Notebook | Experiments | Figures |
|---|---|---|
| `01_gaussian_mixtures.ipynb` | 1 | `exp1_main`, `exp1_ablation` |
| `02_same_mean_shapes.ipynb` | 2 | `exp2_compact_row`, `exp2_ablation_row` |
| `03_movielens_combined.ipynb` | 3 and 4 | `exp3_exp4_combined_row` |

## Repository structure

```
configs/      experiment grids, seeds, calibrated constants (exp1-4.yaml, calibration*.yaml)
data/         MovieLens-1M and the 207-movie pool (gitignored)
docs/         experimental design doc (experiments.md)
notebooks/    figure notebooks (load results only, never recompute)
results/      parquet outputs and figures (gitignored)
scripts/      data download, calibration, per-experiment runners (run_experimentN.py),
              ACB-only passes (run_experiment2_acb.py, run_experiment4_acb.py)
src/kabc/     algorithms, kernels, thresholds, environments, metrics, plotting style
tests/        pytest suite
```

## Citation

```bibtex
@article{thuot2026nonparametric,
  title   = {Nonparametric Kernel Clustering with Bandit Feedback},
  author  = {Thuot, Victor and Vogt, Sebastian and Ghoshdastidar, Debarghya and Verzelen, Nicolas},
  journal = {arXiv preprint arXiv:2601.07535},
  year    = {2026}
}
```