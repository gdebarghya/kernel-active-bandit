"""CLI: python -m scripts.run_experiment3_tasfw_lowbudget [configs/exp3_tasfw_lowbudget.yaml]

Ad hoc follow-up probe, NOT part of the frozen Experiment 3 design in
docs/experiments.md: re-runs ONLY tas-fw, at two FIXED, reduced budget_cap
tiers (0.1x/0.2x of exp3's production budget_cap=15000, i.e. 1500/3000),
across all 50 reps x 3 versions, to check whether tas-fw's point estimate
(sigma_hat) already finds the correct clustering well before it can afford
to formally clear its own beta(t,delta) certification threshold -- prompted
by observing that production tas-fw (results/exp3_*/results.parquet)
almost always reports stopped=False yet still has a near-perfect mean ARI.

WRITES A SEPARATE FILE: results/exp3_tasfw_lowbudget/results.parquet.
Never touches results/exp3_1|2|3/results.parquet (production exp3) or
configs/exp3.yaml -- this is a standalone addition, run independently.

SAME MOVIE PAIRS AS PRODUCTION, BY CONSTRUCTION: this script imports
`_load_movie_pool` and `_choose_pair` directly from scripts/run_experiment3.py
(no reimplementation) and configs/exp3_tasfw_lowbudget.yaml uses the SAME
`exp_id`/`seed_multiplier` as configs/exp3.yaml -- since pair selection
depends only on `rep_seed = seed_multiplier*exp_id + rep` (identical here),
NOT on which algorithms are being fit, every rep/version draws the exact
same real movie pair as the corresponding row in results/exp3_*/results.parquet.
This makes the two tas-fw-0.1x/-0.2x rows directly comparable, instance-for-
instance, to production's own kabc/acb/tas-fw/unif-kkm rows for the same
(rep, version) -- notebooks/03 merges them in on that basis (see its own
comments). Algorithm-level RNG substreams are still independent from
production's own tas-fw row: this script's `variants` list occupies
different `algo_idx` positions in the per-algorithm seed formula than
production's 6-row roster does, so the actual pulls drawn are a fresh,
independent draw, not a truncated prefix of production's tas-fw run.

STRUCTURE: deliberately mirrors run_experiment3.py's own shape (parallelize
across reps via batched joblib `Parallel(n_jobs=n_jobs)`, worker computes
in-memory / main process is the sole parquet writer, incremental per-batch
writes, rep_start-based resume) -- see that script's module docstring for
the full rationale (SANDBOX/WRITER SAFETY notes apply identically here).
Much simpler in one respect: no kabc-derived-variant budget dependency at
all (both variants have their own FIXED budget_cap, independent of any
other row), so there is no cross-algorithm ordering requirement to enforce.

SMOKE_TEST=1: reps capped at 5, same convention as every other dedicated
script (scripts/common.SMOKE_TEST_MAX_REPS).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# BLAS THREAD CAP -- must be set before numpy (or anything importing it) is
# first imported in THIS process. See scripts/common.py's module docstring.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import threadpoolctl
import yaml
from joblib import Parallel, delayed

# SANDBOX: importing scripts.common forces joblib's 'multiprocessing' backend
# process-wide -- see that module's docstring. Also pulls in SMOKE_TEST/
# SMOKE_TEST_MAX_REPS, reused here for the same convention as every other
# dedicated script.
from kabc import metrics
from kabc.algorithms.tas_fw import fit as tas_fw_fit
from kabc.environments import MovieLensPairEnv
from scripts.common import SMOKE_TEST, SMOKE_TEST_MAX_REPS
from scripts.run_experiment3 import _load_movie_pool, _choose_pair  # read-only reuse, see module docstring


def _process_one_rep(rep: int, exp_id: int, seed_multiplier: int, versions: list[dict], variants: list[dict],
                      movie_ids: list[int], means: np.ndarray, titles: dict, pools: dict, mmd2: np.ndarray,
                      arms_per_prototype: int, delta: float) -> dict[str, list[dict]]:
    """Runs every version's every tas-fw budget variant for ONE rep,
    sequentially, entirely in-memory -- see module docstring's WRITER
    SAFETY note (mirrors run_experiment3.py's _process_one_rep exactly in
    shape). Returns {version_name: [row, ...]}.
    """
    rep_seed = seed_multiplier * exp_id + rep
    pair_rng = np.random.default_rng(rep_seed * 100 + 1)
    candidate_order = pair_rng.permutation(len(movie_ids))  # identical draw to run_experiment3.py's own

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

        rows = []
        for variant_idx, variant in enumerate(variants):
            algo_id, budget_cap = variant["id"], variant["budget_cap"]
            # Distinct algo_idx space from production exp3's 6-row roster
            # (module docstring's "independent RNG substream" note) --
            # production's own algo_idx runs 0-5 for a given (rep, version);
            # this script's variant_idx (0-1) is combined the same way but
            # is a completely separate seed stream by construction (same
            # formula shape, but this script is never invoked with
            # production's algo_ids list, so there is no possibility of
            # silently colliding with an existing production seed).
            algo_seed = rep_seed * 1000 + 100 + v_idx * 10 + variant_idx
            rng = np.random.default_rng(algo_seed)
            t0 = time.time()
            with threadpoolctl.threadpool_limits(limits=1):
                result = tas_fw_fit(env, delta, budget_cap, rng)
            wall_clock_seconds = time.time() - t0

            print(f"    rep {rep} {name}/{algo_id}: tau={result.tau} stopped={result.stopped} "
                  f"time={wall_clock_seconds:.1f}s", flush=True)

            row = {
                "exp_id": exp_id,
                "version": name,
                "algorithm": algo_id,
                "rep": rep,
                "seed": algo_seed,
                "budget_cap": budget_cap,
                "tau": result.tau,
                "stopped": result.stopped,
                "wall_clock_seconds": wall_clock_seconds,
                "partition": result.partition.tolist(),
                "error": metrics.error(result.partition, env.true_partition),
                "ari": metrics.ari(result.partition, env.true_partition),
                "n_iterations": len(result.history),
            }
            row.update(pair_info)
            rows.append(row)
        rows_by_version[name] = rows
    return rows_by_version


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/exp3_tasfw_lowbudget.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_id = config["exp_id"]
    delta = config["delta"]
    reps = SMOKE_TEST_MAX_REPS if SMOKE_TEST else config["reps"]
    rep_start = config.get("rep_start", 0)
    seed_multiplier = config["seed_multiplier"]
    data_dir = Path(config["data_dir"])
    arms_per_prototype = config["arms_per_prototype"]
    n_jobs = config.get("n_jobs", 10)
    variants = config["variants"]
    versions = config["versions"]

    movie_ids, means, titles, pools, mmd2 = _load_movie_pool(data_dir)
    n_movies = len(movie_ids)
    n_variant_rows = len(variants)
    rep_indices = list(range(rep_start, reps))
    n_reps_this_run = len(rep_indices)
    print(f"exp3_tasfw_lowbudget: loaded {n_movies} movies from {data_dir}, {n_reps_this_run} reps "
          f"(rep_start={rep_start}, reps={reps}), {len(versions)} versions, "
          f"{n_variant_rows} variant rows ({[v['id'] for v in variants]}) -- "
          f"{n_reps_this_run * len(versions) * n_variant_rows} total runs, "
          f"n_jobs={n_jobs} (parallelizing across reps)", flush=True)

    out_dir = Path("results/exp3_tasfw_lowbudget")
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
                rep, exp_id, seed_multiplier, versions, variants, movie_ids, means, titles, pools, mmd2,
                arms_per_prototype, delta,
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
    print("EXP3_TASFW_LOWBUDGET EXIT CODE: 0")


if __name__ == "__main__":
    main()
