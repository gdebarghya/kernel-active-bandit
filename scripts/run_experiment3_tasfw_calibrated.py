"""CLI: python -m scripts.run_experiment3_tasfw_calibrated [configs/exp3_tasfw_calibrated.yaml]

Ad hoc follow-up (NOT part of the frozen Experiment 3 design): runs ONLY
`tas-fw-calibrated` -- tas-fw with the const from
scripts/calibrate_tas_fw.py's offline, ground-truth-free-at-runtime
calibration (configs/calibration_tas_fw.yaml) -- at Exp3's REAL production
budget_cap=15000, across all 50 reps x 3 versions, to check whether a
calibrated threshold gets tas-fw to actually certify more often on real
data (not just whether cutting the budget changes its point estimate, which
scripts/run_experiment3_tasfw_lowbudget.py already checked).

WRITES A SEPARATE FILE: results/exp3_tasfw_calibrated/results.parquet.
Never touches results/exp3_1|2|3/results.parquet (production) or
results/exp3_tasfw_lowbudget/results.parquet (the earlier probe).

SAME MOVIE PAIRS AS PRODUCTION, BY CONSTRUCTION: same mechanism as
scripts/run_experiment3_tasfw_lowbudget.py -- imports `_load_movie_pool`/
`_choose_pair` read-only from scripts/run_experiment3.py, and
configs/exp3_tasfw_calibrated.yaml uses the SAME exp_id/seed_multiplier as
configs/exp3.yaml, so every rep/version draws the exact same real movie
pair as the corresponding row in results/exp3_*/results.parquet -- directly
comparable, instance-for-instance, to production's own `tas-fw` row.

CALIBRATED CONST: read once at import time from
configs/calibration_tas_fw.yaml (written by `python -m
scripts.calibrate_tas_fw`) -- fails loudly if that file doesn't exist yet
(calibration must run first), same "fail loudly, point to the generating
command" convention as every other missing-results case in this repo.

STRUCTURE: mirrors run_experiment3_tasfw_lowbudget.py's own shape (batched
joblib Parallel(n_jobs=n_jobs) across reps, sole-writer main process,
rep_start-based resume) -- see that script's module docstring for the full
rationale. No cross-algorithm budget dependency here either (one algorithm
row, one fixed const, independent of any other row).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import threadpoolctl
import yaml
from joblib import Parallel, delayed

from kabc import metrics
from kabc.algorithms.tas_fw import _fit as tas_fw_fit_calibrated
from kabc.environments import MovieLensPairEnv
from scripts.common import SMOKE_TEST, SMOKE_TEST_MAX_REPS
from scripts.run_experiment3 import _load_movie_pool, _choose_pair  # read-only reuse

CALIBRATED_CONST_PATH = Path("configs/calibration_tas_fw.yaml")
ALGO_ID = "tas-fw-calibrated"


def _load_calibrated_const() -> float:
    if not CALIBRATED_CONST_PATH.exists():
        raise FileNotFoundError(
            f"{CALIBRATED_CONST_PATH} not found. Run `python -m scripts.calibrate_tas_fw` first "
            "to freeze a calibrated const -- this script only consumes it, it never recomputes it."
        )
    with open(CALIBRATED_CONST_PATH) as f:
        cfg = yaml.safe_load(f)
    return float(cfg["const"])


CONST = _load_calibrated_const()


def _process_one_rep(rep: int, exp_id: int, seed_multiplier: int, versions: list[dict],
                      movie_ids: list[int], means: np.ndarray, titles: dict, pools: dict, mmd2: np.ndarray,
                      arms_per_prototype: int, budget_cap: int, delta: float, const: float) -> dict[str, list[dict]]:
    """Runs every version's tas-fw-calibrated row for ONE rep, in-memory --
    mirrors run_experiment3_tasfw_lowbudget.py's own function exactly, one
    algorithm row instead of two."""
    rep_seed = seed_multiplier * exp_id + rep
    pair_rng = np.random.default_rng(rep_seed * 100 + 1)
    candidate_order = pair_rng.permutation(len(movie_ids))

    rows_by_version: dict[str, list[dict]] = {}
    for v_idx, version in enumerate(versions):
        name = version["name"]
        a_idx, b_idx = _choose_pair(version, v_idx, rep_seed, candidate_order, means, mmd2)
        mid_a, mid_b = movie_ids[a_idx], movie_ids[b_idx]
        pair_info = {
            "movie_id_a": mid_a,
            "movie_id_b": mid_b,
            "title_a": titles[mid_a],
            "title_b": titles[mid_b],
            "mean_a": float(means[a_idx]),
            "mean_b": float(means[b_idx]),
            "mmd2_ab": float(mmd2[a_idx, b_idx]),
        }
        env = MovieLensPairEnv(pools[mid_a], pools[mid_b], arms_per_prototype, mid_a, mid_b, titles[mid_a], titles[mid_b])

        # Distinct algo_idx (0) from both production exp3's roster (0-5) and
        # the lowbudget probe's own (0-1) -- an independent RNG substream,
        # same convention as run_experiment3_tasfw_lowbudget.py's own note.
        algo_seed = rep_seed * 1000 + 100 + v_idx * 10 + 0
        rng = np.random.default_rng(algo_seed)
        t0 = time.time()
        with threadpoolctl.threadpool_limits(limits=1):
            result = tas_fw_fit_calibrated(env, delta, budget_cap, rng, const=const)
        wall_clock_seconds = time.time() - t0

        print(f"    rep {rep} {name}/{ALGO_ID}: tau={result.tau} stopped={result.stopped} "
              f"time={wall_clock_seconds:.1f}s", flush=True)

        row = {
            "exp_id": exp_id,
            "version": name,
            "algorithm": ALGO_ID,
            "rep": rep,
            "seed": algo_seed,
            "budget_cap": budget_cap,
            "const": const,
            "tau": result.tau,
            "stopped": result.stopped,
            "wall_clock_seconds": wall_clock_seconds,
            "partition": result.partition.tolist(),
            "error": metrics.error(result.partition, env.true_partition),
            "ari": metrics.ari(result.partition, env.true_partition),
            "n_iterations": len(result.history),
        }
        row.update(pair_info)
        rows_by_version.setdefault(name, []).append(row)
    return rows_by_version


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/exp3_tasfw_calibrated.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_id = config["exp_id"]
    delta = config["delta"]
    reps = SMOKE_TEST_MAX_REPS if SMOKE_TEST else config["reps"]
    rep_start = config.get("rep_start", 0)
    seed_multiplier = config["seed_multiplier"]
    data_dir = Path(config["data_dir"])
    arms_per_prototype = config["arms_per_prototype"]
    budget_cap = config["budget_cap"]
    n_jobs = config.get("n_jobs", 10)
    versions = config["versions"]

    movie_ids, means, titles, pools, mmd2 = _load_movie_pool(data_dir)
    n_movies = len(movie_ids)
    rep_indices = list(range(rep_start, reps))
    n_reps_this_run = len(rep_indices)
    print(f"exp3_tasfw_calibrated: loaded {n_movies} movies from {data_dir}, const={CONST:.6f} "
          f"(from {CALIBRATED_CONST_PATH}), {n_reps_this_run} reps (rep_start={rep_start}, reps={reps}), "
          f"{len(versions)} versions, budget_cap={budget_cap}, n_jobs={n_jobs}", flush=True)

    out_dir = Path("results/exp3_tasfw_calibrated")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.parquet"

    all_rows: list[dict] = []
    if rep_start > 0 and out_path.exists():
        existing_df = pd.read_parquet(out_path)
        all_rows = existing_df.to_dict("records")
        print(f"resume: loaded {len(all_rows)} existing rows from {out_path}", flush=True)

    t_start = time.time()
    n_done = 0
    for batch_start in range(0, len(rep_indices), n_jobs):
        batch = rep_indices[batch_start : batch_start + n_jobs]
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(_process_one_rep)(
                rep, exp_id, seed_multiplier, versions, movie_ids, means, titles, pools, mmd2,
                arms_per_prototype, budget_cap, delta, CONST,
            )
            for rep in batch
        )
        for rows_by_version in batch_results:
            for rows in rows_by_version.values():
                all_rows.extend(rows)
        pd.DataFrame(all_rows).to_parquet(out_path, index=False)
        n_done += len(batch)
        elapsed = time.time() - t_start
        print(f"{n_done}/{n_reps_this_run} reps done, elapsed={elapsed:.1f}s", flush=True)

    print(f"Wrote {len(all_rows)} rows to {out_path}", flush=True)
    print("EXP3_TASFW_CALIBRATED EXIT CODE: 0")


if __name__ == "__main__":
    main()
