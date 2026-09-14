# kernel-active-bandit

Benchmark suite for **KABC** (Kernel-based Active Clustering of Bandit Arms) —
a nonparametric method for actively clustering bandit arms based on their full
distributions, not just means. Companion code for our paper (arXiv 2601.07535).

Compares KABC against parametric baselines (ACB, TaS-FW) and a uniform-sampling
baseline, on synthetic Gaussian mixtures and real MovieLens-1M rating data.

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
make movielens-207  # build data/ml-1m-207/ (the >=1000-rating movie pool exp3 samples pairs from)
make calibrate       # freeze practical threshold constants
make smoke           # quick sanity run, experiments 1-2, reduced grid
make smoke-exp3      # separate, slower sanity run for experiment 3 (includes tas-fw)
make exp1            # Experiment 1: Gaussian mixtures
make exp2            # Experiment 2: same-mean, different-shape instances
make exp3            # Experiment 3: MovieLens random pairs
make test            # run test suite
```

Notebooks in `notebooks/` load results from `results/*.parquet` and produce
paper figures — run the corresponding `make expN` first.

## Repository structure

configs/      experiment grids, seeds, calibrated constants
data/         MovieLens-1M (gitignored)
docs/         experimental design doc
notebooks/    one notebook per experiment, figures only
results/      parquet outputs (gitignored)
scripts/      data download, calibration, experiment runner
src/kabc/     algorithms, kernels, thresholds, environments, metrics
tests/        pytest suite

## Citation

```bibtex
@article{thuot2026nonparametric,
  title   = {Nonparametric Kernel Clustering with Bandit Feedback},
  author  = {Thuot, Victor and Vogt, Sebastian and Ghoshdastidar, Debarghya and Verzelen, Nicolas},
  journal = {arXiv preprint arXiv:2601.07535},
  year    = {2026}
}
```