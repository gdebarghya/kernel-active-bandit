"""CLI: python -m scripts.run_experiment3 [configs/exp3.yaml]

Runs Experiment 3 (docs/experiments.md, "Experiment 3" section): three
versions (exp3_1/exp3_2/exp3_3) of a MovieLens random-pair robustness check.
Saves THREE separate files, one per version -- results/exp3_1/results.parquet,
results/exp3_2/results.parquet, results/exp3_3/results.parquet -- so
notebooks/03's plotting code can load and process one version at a time with
the same logic.

RENUMBERING NOTE: this experiment was originally built and run as
"Experiment 4" (MovieLens random pairs), alongside a separate, older
"Experiment 3" (one fixed, cherry-picked K=4 prototype instance, plus a small
N=6 reduced instance built just for tas-fw). That old Experiment 3 has been
DROPPED ENTIRELY -- the whole point of keeping MovieLens in this benchmark is
testing against tas-fw on real data, which this random-pair design already
does, more robustly (averaged over many sampled pairs rather than one
cherry-picked instance) than the old fixed instance did. This script took the
"3" slot. See docs/experiments.md's REVISION 4 note for the full rationale.

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

PARALLELIZATION: across REPS, not across the 6 (algorithm) rows within one
rep. `rep_indices` are chunked into `n_jobs`-sized batches; each batch runs
one blocking `Parallel(n_jobs=n_jobs)` call over `_process_one_rep`, so every
batch's reps run truly concurrently across `n_jobs` worker processes. INSIDE
one call, all 3 versions and all 6 algorithm rows per version run
SEQUENTIALLY in a single worker process. Two reasons this beats
parallelizing the 18 (version, algorithm) items within one rep: (1) it
resolves the kabc-derived-variant budget dependency below for free -- running
kabc before its derived rows in the same worker just means calling it first,
no separate dependency-ordering phase needed; (2) better load balancing
within a batch -- tas-fw-heavy vs. cheap reps land across workers by chance,
rather than every single rep's wall-clock being gated by its slowest
algorithm. Progress/incremental writes happen once per BATCH (not per single
rep as it completes) -- see the SANDBOX note below for why.
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

ALGORITHM ROSTER (6 rows per (rep, version)): kabc, kabc-0.5x, acb, tas-fw,
unif-kkm-1x, unif-kkm-0.5x. `kabc-sg` and `unif-kkm-2x` (both present in
this experiment's original "Experiment 4" run) are DROPPED: kabc-sg's
fragility / confident-wrong-answer failure mode is already thoroughly
documented on this exact real-data setting from that prior run, and
unif-kkm-2x never showed much beyond what -1x already showed. See
docs/experiments.md's REVISION 4 note.

KABC-0.5X VARIANT (renamed from kabc-0.9x, 2026-09-14): a second,
independent `kabc` run per (rep, version), at `budget_cap = min(0.5 *
tau_kabc, budget_cap)` where `tau_kabc` is that SAME (rep, version)'s
primary `kabc` row's own realized tau. This asks: how conservative is
kabc's stopping rule -- what does its partition estimate look like with a
meaningfully smaller budget than it actually decided it needed? Uses a
FRESH, independent RNG substream (its own `algo_seed`, same convention as
every other algorithm row), not the primary kabc run's exact same pulls
truncated early -- deliberately: reusing the same seed was considered, but
decided against, since results here are averaged over many reps anyway, so
an independent draw at the reduced budget is simpler and equally
informative in aggregate. `kabc` must therefore run before `kabc-0.5x` for
a given (rep, version) -- same ordering requirement as the unif-kkm
variants below, and enforced the same way.

MULTIPLIER CHANGED from an earlier 0.9x to 0.5x (2026-09-14) -- same
reasoning as Experiments 1-2's dedicated scripts: KABC's doubling-trick
schedule quantizes its own realized tau to a small set of round-boundary
checkpoints, and the FINAL round alone typically costs roughly half of the
total realized tau, so a 0.9x cut cannot actually test "10% less budget" --
it almost always lands in the same "one checkpoint short" bucket as a much
larger cut would. 0.5x is honest about what is actually being tested: a
meaningfully smaller, still-real budget cut.

UNIF-KKM VARIANTS (docs/experiments.md's "unif-kkm variants"): unif-kkm is
non-adaptive and always spends whatever budget it's given
(`baselines.fit`'s docstring: n = budget_cap // N pulls/arm, always) --
passing it a smaller budget_cap than the configured value is already the
built-in mechanism for "give it fewer samples", no code change to
baselines.py needed. unif-kkm-1x and unif-kkm-0.5x use that SAME (rep,
version)'s kabc tau, at a 1.0x/0.5x multiple (capped at budget_cap) -- so
kabc's row must be computed before either, for a given (rep, version).
`main()` forces kabc first in the algorithm order regardless of the config
file's own ordering (see its "kabc must resolve first" block below); this
also covers kabc-0.5x, which has the identical dependency.

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
from kabc.algorithms.tas_fw import fit as tas_fw_fit
from kabc.environments import MovieLensPairEnv, _movie_rating_distributions, _prototype_bandwidth, compute_pairwise_mmd2
from scripts.common import SMOKE_TEST, SMOKE_TEST_MAX_REPS, _acb_fit, _kabc_fit

# kabc-0.5x and unif-kkm-1x/0.5x are NOT looked up in ALGORITHMS below --
# they're resolved specially in _process_one_rep once kabc's tau for that
# (rep, version) is known (see module docstring's "KABC-0.5X VARIANT" and
# "UNIF-KKM VARIANTS"). _kabc_fit/unif_kkm_fit are still imported/used
# directly for those paths, not through this dict.
ALGORITHMS = {
    "kabc": _kabc_fit,
    "acb": _acb_fit,
    "tas-fw": tas_fw_fit,
}
KABC_TAU_MULTIPLE_VARIANTS = {"kabc-0.5x": 0.5}
UNIF_KKM_TAU_MULTIPLE_VARIANTS = {"unif-kkm-1x": 1.0, "unif-kkm-0.5x": 0.5}
TAU_KABC_DEPENDENT_VARIANTS = set(KABC_TAU_MULTIPLE_VARIANTS) | set(UNIF_KKM_TAU_MULTIPLE_VARIANTS)


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
        if algo_id in KABC_TAU_MULTIPLE_VARIANTS:
            this_budget = int(min(KABC_TAU_MULTIPLE_VARIANTS[algo_id] * tau_kabc, budget_cap))
            return _kabc_fit(env, delta, this_budget, rng), this_budget
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

    # kabc must resolve before any tau_kabc-dependent variant (kabc-0.5x,
    # unif-kkm-1x/0.5x -- see module docstring) -- enforced here regardless
    # of the config file's own algorithm ordering, rather than trusting the
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
