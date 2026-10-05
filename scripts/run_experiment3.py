"""CLI: python -m scripts.run_experiment3 [configs/exp3.yaml]

Runs Experiment 3 (docs/experiments.md, "Experiment 3" section): a
MovieLens random-pair robustness check. Saves one results file per
version listed in configs/exp3.yaml's `versions` -- currently just one,
`exp3_3` (results/exp3_3/results.parquet) -- the script itself still
supports multiple versions per run (loops over `config["versions"]`,
writes one file per entry), unchanged from when it ran exp3_1/exp3_2/exp3_3
together (VERSIONS TRIMMED, 2026-10-03: down to exp3_3 only, a paper-space
constraint -- see configs/exp3.yaml's own note for why exp3_3 specifically
was kept). notebooks/03's plotting code loads and processes one version at
a time with the same logic regardless of how many are configured.

RENUMBERING NOTE: this experiment was originally built and run as
"Experiment 4" (MovieLens random pairs), alongside a separate, older
"Experiment 3" (one fixed, cherry-picked K=4 prototype instance, plus a small
N=6 reduced instance built just for tas-fw). That old Experiment 3 has been
DROPPED ENTIRELY -- the whole point of keeping MovieLens in this benchmark is
testing against tas-fw on real data, which this random-pair design already
does, more robustly (averaged over many sampled pairs rather than one
cherry-picked instance) than the old fixed instance did. This script took the
"3" slot.

WHY A SEPARATE SCRIPT: every experiment 1-2 grid point builds ONE
environment, reused across all its reps (only pull-noise varies by rep).
Experiment 3 instead builds a FRESH environment every single rep (a newly,
randomly sampled movie pair) -- a structurally different loop shape from
either of those. It DOES reuse the shared calibration-loading and
calibrated-fit wrappers directly (`_kabc_fit`/`_acb_fit`, from
`scripts/common.py` -- see that module's docstring) so Experiment 3 draws
on the exact same frozen configs/calibration.yaml constants as Experiments
1-2, per the "calibrate once, reuse everywhere" protocol (section 0.2) --
not a re-implementation.

PARALLELIZATION: across REPS, not across the 4 (algorithm) rows within one
rep. `rep_indices` are chunked into `n_jobs`-sized batches; each batch runs
one blocking `Parallel(n_jobs=n_jobs)` call over `_process_one_rep`, so every
batch's reps run truly concurrently across `n_jobs` worker processes. INSIDE
one call, all configured versions (currently 1) and all 4 algorithm rows
per version run SEQUENTIALLY in a single worker process. Two reasons this
beats parallelizing the (version, algorithm) items within one rep: (1) it
resolves `unif-kkm`'s budget dependency on kabc's own tau for free -- running
kabc before it in the same worker just means calling it first, no separate
dependency-ordering phase needed; (2) better load balancing within a batch
-- tas-fw-heavy vs. cheap reps land across workers by chance, rather than
every single rep's wall-clock being gated by its slowest algorithm.
Progress/incremental writes happen once per BATCH (not per single rep as it
completes) -- see the SANDBOX note below for why.
WRITER SAFETY: workers never touch the filesystem -- `_process_one_rep`
computes entirely in-memory and returns that rep's rows; only the MAIN
process (after each batch) calls `to_parquet(...)`, always sequentially, one
batch's results at a time. Exactly one writer, same as run_experiment1.py's
worker-computes/main-process-writes pattern -- no concurrent-write risk.

SANDBOX: this repo's dev sandbox blocks a syscall (`os.sysconf`) that
joblib's default `loky` backend needs at worker-pool startup, so
`joblib.parallel_config(backend="multiprocessing")` is forced near the top
of this file (see that note there). `multiprocessing` doesn't support
`return_as="generator"` (per-rep streaming results -- reps completing OUT OF
ORDER across workers, written to parquet as soon as each one lands) -- only
`loky`/`dask` do. Batching into blocking per-`n_jobs`-reps `Parallel()` calls
(this function's actual current shape) is the closest equivalent under that
constraint: still real multi-process parallelism, still periodic progress
printing and incremental parquet writes (crash-resume safety preserved),
just at batch granularity instead of single-rep-as-it-completes granularity.

ALGORITHM ROSTER (4 rows per (rep, version), REVISION 2026-10-03): kabc,
acb, tas-fw, unif-kkm. Replaces the prior 6-row roster (kabc, kabc-0.5x,
acb, tas-fw, unif-kkm-1x, unif-kkm-0.5x) with two changes:

1. **`acb` UNCHANGED -- kept.** Considered for removal as a poor baseline
   on this setting, but it's also consistently cheap here (sub-second/rep,
   real-measured across all three versions of the prior 3-version run),
   so there was no compute reason to drop it either.
2. **Uncalibrated `tas-fw` DROPPED; `tas-fw-calibrated` PROMOTED to the
   canonical `tas-fw`.** The uncalibrated version almost always hit
   `budget_cap` without stopping (despite good ARI on the partition it
   DID find) -- see docs/experiments.md's Experiment 3 section.
   The ad hoc follow-up `scripts/run_experiment3_tasfw_calibrated.py`
   (now DELETED, its one row folded in here) showed a calibrated const
   (`configs/calibration_tas_fw.yaml`, from `scripts/calibrate_tas_fw.py`'s
   own offline protocol -- NOT `configs/calibration.yaml`'s kabc/kabc-sg/acb
   constants) gets tas-fw to actually certify far more often on the SAME
   real movie pairs, at the SAME production `budget_cap`. This script now
   reads that calibrated const directly (`_load_tas_fw_const`) and reports
   it AS `tas-fw` -- not a separate `-calibrated` suffix -- since the
   uncalibrated version is no longer run at all; there is only one `tas-fw`
   row now, and it is the calibrated one.

NO CALIBRATION-CONSTANT ABLATION HERE (revised again, same day): an
earlier pass added `kabc-2x`/`kabc-0.5x`/`tas-fw-2x`/`tas-fw-0.5x`
(mirroring Experiments 1-2's constant ablation), but `tas-fw` is slow
enough (its per-time-step Frank-Wolfe LP cost compounds with however many
samples it ends up pulling before stopping) that running THREE tas-fw
variants per rep wasn't worth it -- dropped entirely, not kept as an
appendix-only row. `kabc-0.5x`/`kabc-2x` were dropped alongside it for
consistency (no ablation at all here, rather than an asymmetric kabc-only
one). `_kabc_fit_0p5x`/`_kabc_fit_2x` (from `scripts/common.py`) remain
used by Experiments 1-2, just not here.

UNIF-KKM (renamed from `unif-kkm-1x`; `unif-kkm-0.5x` DROPPED, mirroring
Experiments 1-2): the only row left with a cross-algorithm tau dependency.
unif-kkm is non-adaptive and always spends whatever budget it's given
(`baselines.fit`'s docstring: n = budget_cap // N pulls/arm, always) --
passing it a smaller budget_cap than the configured value is already the
built-in mechanism for "give it fewer samples", no code change to
baselines.py needed. `unif-kkm` uses that SAME (rep, version)'s kabc tau,
at a 1.0x multiple (capped at budget_cap) -- so kabc's row must be
computed first, for a given (rep, version). `main()` forces kabc first in
the algorithm order regardless of the config file's own ordering (see its
"kabc must resolve first" block below).

VERSIONS TRIMMED (2026-10-03, configs/exp3.yaml): from 3 (exp3_1, exp3_2,
exp3_3) down to exactly 1 (exp3_3 only) -- a paper-space constraint, not a
code change. exp3_3 was kept as the hardest of the three (uniform
B-selection; mean_gap_tol widened from 0.25 to 1.0 on 2026-10-03) where both kabc and the now-
calibrated tas-fw still perform well. See that config's own note for the
full rationale; this script's version-loop machinery is unchanged and
still supports more than one version if one is added back later.

SMOKE_TEST=1: reps capped at 5. Does not shrink N/K (already tiny: N=8) or
the movie pool.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# BLAS THREAD CAP -- must be set before numpy (or anything importing it,
# including scripts.common below) is first imported in THIS process. See
# scripts/common.py's module docstring for why this can't be delegated
# there and the full oversubscription rationale.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import threadpoolctl
import yaml
from joblib import Parallel, delayed

# SANDBOX: importing scripts.common forces joblib's 'multiprocessing'
# backend process-wide (works around a syscall this repo's dev sandbox
# blocks in joblib's default 'loky' backend) -- see that module's docstring
# for the full detail.
from kabc import metrics
from kabc.algorithms.baselines import fit as unif_kkm_fit
from kabc.algorithms.tas_fw import _fit as _tas_fw_fit_internal
from kabc.environments import MovieLensPairEnv, _movie_rating_distributions, _prototype_bandwidth, compute_pairwise_mmd2
from scripts.common import SMOKE_TEST, SMOKE_TEST_MAX_REPS, _acb_fit, _kabc_fit

# TAS-FW CALIBRATED CONST: read once at import time from
# configs/calibration_tas_fw.yaml (written by `python -m
# scripts.calibrate_tas_fw`), NOT configs/calibration.yaml -- tas-fw's
# calibration protocol is structurally different (offline, synthetic
# DiscretePMFPairEnv instances, no live budget_cap-dependent search against
# this experiment's own data) and has always lived in its own file -- see
# that config's header comment and scripts/calibrate_tas_fw.py's module
# docstring. Fails loudly if calibration hasn't been run yet, same
# "fail loudly, point to the generating command" convention as every other
# missing-results case in this repo.
TAS_FW_CALIBRATION_PATH = Path("configs/calibration_tas_fw.yaml")


def _load_tas_fw_const() -> float:
    if not TAS_FW_CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"{TAS_FW_CALIBRATION_PATH} not found. Run `python -m scripts.calibrate_tas_fw` first "
            "to freeze a calibrated const -- this script only consumes it, it never recomputes it."
        )
    with open(TAS_FW_CALIBRATION_PATH) as f:
        return float(yaml.safe_load(f)["const"])


_TAS_FW_CONST = _load_tas_fw_const()


def _tas_fw_fit(env, delta, budget_cap, rng):
    return _tas_fw_fit_internal(env, delta, budget_cap, rng, const=_TAS_FW_CONST)


# unif-kkm is NOT looked up in ALGORITHMS below -- it's resolved specially
# in _process_one_rep once kabc's tau for that (rep, version) is known (see
# module docstring's "UNIF-KKM VARIANT"). unif_kkm_fit is still
# imported/used directly for that path, not through this dict.
ALGORITHMS = {
    "kabc": _kabc_fit,
    "acb": _acb_fit,
    "tas-fw": _tas_fw_fit,
}
UNIF_KKM_TAU_MULTIPLE_VARIANTS = {"unif-kkm": 1.0}
TAU_KABC_DEPENDENT_VARIANTS = set(UNIF_KKM_TAU_MULTIPLE_VARIANTS)


def _load_movie_pool(data_dir: Path):
    """Loads scripts/build_movielens_207.py's output. Returns:
    movie_ids (list, in a fixed row/column order shared by every array
    below), means (np.ndarray), titles (dict), pools (dict: movie_id ->
    ratings np.ndarray), mmd2 (np.ndarray, (M,M), aligned to movie_ids).
    """
    movies = pd.read_parquet(data_dir / "movies.parquet")
    ratings = pd.read_parquet(data_dir / "ratings.parquet")

    probs, means_dict = _movie_rating_distributions(ratings, min_ratings=1000)
    bandwidth = _prototype_bandwidth(ratings, probs.keys(), sample_size=5000, seed=0)
    movie_ids, mmd2 = compute_pairwise_mmd2(probs, bandwidth)

    titles = dict(zip(movies["movie_id"], movies["title"]))
    means = np.array([means_dict[mid] for mid in movie_ids])
    pools = {mid: g["rating"].to_numpy(dtype=float) for mid, g in ratings.groupby("movie_id")}
    return movie_ids, means, titles, pools, mmd2


def _eligible_mask(comparison: str, version: dict, gaps: np.ndarray) -> np.ndarray:
    """Boolean mask over `gaps` (|mean_A - mean_B| for every candidate movie)
    per `comparison`: `le` = gap <= mean_gap_tol (one-sided); `range` =
    mean_gap_min <= gap <= mean_gap_max (bounded both sides, e.g. exp3_2 --
    excludes both near-zero-gap and trivially-easy very-large-gap pairs).
    """
    if comparison == "le":
        return gaps <= version["mean_gap_tol"]
    if comparison == "range":
        return (gaps >= version["mean_gap_min"]) & (gaps <= version["mean_gap_max"])
    raise ValueError(f"unknown comparison {comparison!r} -- expected 'le' or 'range'")


def _choose_pair(version: dict, v_idx: int, rep_seed: int, candidate_order: np.ndarray, means: np.ndarray, mmd2: np.ndarray):
    """Picks (a_idx, b_idx) into the shared movie_ids/means/mmd2 arrays for
    one (rep, version). `candidate_order` is a permutation of movie indices,
    identical across all three versions for a given rep (common random
    numbers) -- this function scans it in order and returns the FIRST
    candidate A with >= 1 eligible B under THIS version's rule, so versions
    only diverge on A when the shared first choice has no eligible partner
    under a stricter rule (see configs/exp3.yaml's versions comment).
    """
    comparison, rule = version["comparison"], version["rule"]
    b_rng = np.random.default_rng(rep_seed * 100 + 10 + v_idx)

    for a_idx in candidate_order:
        gaps = np.abs(means - means[a_idx])
        gaps[a_idx] = np.inf  # exclude self
        eligible = np.where(_eligible_mask(comparison, version, gaps))[0]
        if len(eligible) == 0:
            continue
        if rule == "max_mmd":
            b_idx = int(eligible[np.argmax(mmd2[a_idx, eligible])])
        else:
            b_idx = int(eligible[b_rng.integers(len(eligible))])
        return int(a_idx), b_idx
    raise RuntimeError(
        f"version {version['name']!r}: no movie in the dataset has an eligible partner "
        f"under comparison={comparison!r} -- check the config"
    )


def _fit_one_algorithm(algo_id: str, env, delta: float, budget_cap: int, rng: np.random.Generator, tau_kabc: float | None):
    if algo_id in TAU_KABC_DEPENDENT_VARIANTS:
        if tau_kabc is None:
            raise RuntimeError(f"{algo_id} requires kabc's tau to already be known for this (rep, version) -- "
                                f"check that 'kabc' precedes it in configs/exp3.yaml's algorithms list")
        this_budget = int(min(UNIF_KKM_TAU_MULTIPLE_VARIANTS[algo_id] * tau_kabc, budget_cap))
        return unif_kkm_fit(env, delta, this_budget, rng), this_budget
    fit_fn = ALGORITHMS[algo_id]
    return fit_fn(env, delta, budget_cap, rng), budget_cap


def _process_one_rep(rep: int, exp_id: int, seed_multiplier: int, versions: list[dict], algo_ids: list[str],
                      movie_ids: list[int], means: np.ndarray, titles: dict, pools: dict, mmd2: np.ndarray,
                      arms_per_prototype: int, budget_cap: int, delta: float) -> dict[str, list[dict]]:
    """Runs every version's every algorithm row for ONE rep, sequentially,
    entirely in-memory (no filesystem access) -- see module docstring's
    PARALLELIZATION note. Returns {version_name: [row, ...]}.
    """
    rep_seed = seed_multiplier * exp_id + rep
    pair_rng = np.random.default_rng(rep_seed * 100 + 1)
    candidate_order = pair_rng.permutation(len(movie_ids))  # shared A-candidate order, all versions

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
        tau_kabc = None
        for algo_idx, algo_id in enumerate(algo_ids):
            algo_seed = rep_seed * 1000 + 100 + v_idx * 10 + algo_idx
            rng = np.random.default_rng(algo_seed)
            t0 = time.time()
            with threadpoolctl.threadpool_limits(limits=1):
                result, used_budget_cap = _fit_one_algorithm(algo_id, env, delta, budget_cap, rng, tau_kabc)
            wall_clock_seconds = time.time() - t0
            if algo_id == "kabc":
                tau_kabc = result.tau

            # Per-worker, per-algorithm progress -- printed from INSIDE the
            # worker process, not just aggregated in the main process after
            # a whole rep completes. Forked ('multiprocessing' backend)
            # worker stdout is inherited by the main process when the whole
            # script's output is redirected to a file (the same mechanism
            # that already surfaces workers' own UserWarnings in every
            # dedicated script's logs), so this gives live visibility into
            # what all `n_jobs` workers are doing concurrently, not just a
            # completed-rep counter.
            print(f"    rep {rep} {name}/{algo_id}: tau={result.tau} stopped={result.stopped} "
                  f"time={wall_clock_seconds:.1f}s", flush=True)

            row = {
                "exp_id": exp_id,
                "version": name,
                "algorithm": algo_id,
                "rep": rep,
                "seed": algo_seed,
                "budget_cap": used_budget_cap,
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
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/exp3.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_id = config["exp_id"]
    delta = config["delta"]
    reps = SMOKE_TEST_MAX_REPS if SMOKE_TEST else config["reps"]
    rep_start = config.get("rep_start", 0)  # resume support: skip [0, rep_start) --
                                             # e.g. after a crash on one specific rep,
                                             # rather than re-running (and re-crashing
                                             # on) already-completed reps. Default 0
                                             # preserves the original full-range behavior.
    seed_multiplier = config["seed_multiplier"]
    data_dir = Path(config["data_dir"])
    arms_per_prototype = config["arms_per_prototype"]
    budget_cap = config["budget_cap"]
    n_jobs = config.get("n_jobs", 10)
    algo_ids = config["algorithms"]
    versions = config["versions"]

    # kabc must resolve before unif-kkm (the only tau_kabc-dependent row
    # left -- see module docstring) -- enforced here regardless of the
    # config file's own algorithm ordering, rather than trusting the
    # config to always list kabc first.
    if "kabc" in algo_ids and any(a in TAU_KABC_DEPENDENT_VARIANTS for a in algo_ids):
        algo_ids = ["kabc"] + [a for a in algo_ids if a != "kabc"]

    movie_ids, means, titles, pools, mmd2 = _load_movie_pool(data_dir)
    n_movies = len(movie_ids)
    n_algo_rows = len(algo_ids)
    rep_indices = list(range(rep_start, reps))
    n_reps_this_run = len(rep_indices)
    print(f"exp3: loaded {n_movies} movies from {data_dir}, {n_reps_this_run} reps "
          f"(rep_start={rep_start}, reps={reps}), {len(versions)} versions, "
          f"{n_algo_rows} algorithm rows -- {n_reps_this_run * len(versions) * n_algo_rows} total runs, "
          f"n_jobs={n_jobs} (parallelizing across reps)", flush=True)

    for version in versions:
        Path(f"results/{version['name']}").mkdir(parents=True, exist_ok=True)

    # Seed all_rows from any EXISTING results (resume support): rep_start > 0
    # means earlier reps were already run and saved in a prior invocation --
    # load them so the incremental writer below produces a complete file
    # (old reps + new reps), not one that silently drops the old reps.
    all_rows = {}
    for v in versions:
        existing_path = Path(f"results/{v['name']}/results.parquet")
        if rep_start > 0 and existing_path.exists():
            existing_df = pd.read_parquet(existing_path)
            all_rows[v["name"]] = existing_df.to_dict("records")
            print(f"resume: loaded {len(all_rows[v['name']])} existing rows from {existing_path}", flush=True)
        else:
            all_rows[v["name"]] = []
    t_start = time.time()
    n_done = 0

    # Batched (not return_as="generator") -- see module docstring's SANDBOX
    # note: 'multiprocessing' (forced above) doesn't support generator
    # streaming, only 'loky'/'dask' do. Each batch is still a real
    # n_jobs-way concurrent Parallel() call.
    for batch_start in range(0, len(rep_indices), n_jobs):
        batch = rep_indices[batch_start : batch_start + n_jobs]
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(_process_one_rep)(
                rep, exp_id, seed_multiplier, versions, algo_ids, movie_ids, means, titles, pools, mmd2,
                arms_per_prototype, budget_cap, delta,
            )
            for rep in batch
        )
        for rows_by_version in batch_results:
            for name, rows in rows_by_version.items():
                all_rows[name].extend(rows)
        for version in versions:
            df = pd.DataFrame(all_rows[version["name"]])
            df.to_parquet(f"results/{version['name']}/results.parquet", index=False)
        n_done += len(batch)
        elapsed = time.time() - t_start
        print(f"{n_done}/{n_reps_this_run} reps done, elapsed={elapsed:.1f}s", flush=True)

    for version in versions:
        n_rows = len(all_rows[version["name"]])
        print(f"Wrote {n_rows} rows to results/{version['name']}/results.parquet", flush=True)
    print("EXP3 EXIT CODE: 0")


if __name__ == "__main__":
    main()
