"""CLI: python -m scripts.run_experiment4 [configs/exp4.yaml]

Runs Experiment 4 (docs/experiments.md, "Experiment 4" section): does kabc's
decision rule (pairwise threshold + connected components) hold up better than
unif-kkm's k-means under real, heterogeneous MovieLens difficulty and
cluster-size imbalance? Two settings, each its own results file
(results/exp4_{setting}/results.parquet):

- `baseline`: K=7, same role/count structure the earlier design's
  `resampled` condition used, budget_cap raised to clear k=8's full
  cumulative cost (removes the earlier cap-driven truncation at k=7).
- `amplified_imbalance`: K=12, a much more skewed far-pair ratio (20:2, not
  12:2) -- the regime where the earlier design's `track_a` showed the
  cleanest imbalance-specific signal, stressed harder here.

CLEAN REDESIGN (2026-09-16), replacing the earlier 4-condition
(fixed/resampled/track_a/track_b) x 6-algorithm version entirely -- that
version's own results/notebook/docs sections are superseded, not merged.
Two algorithms only (kabc, unif-kkm-1x, matched-budget) -- see
configs/exp4.yaml's own module comment for the full rationale, including the
real (not just formula) memory-guard verification that motivated dropping
the hand-picked `fixed` instance and the `track_a`/`track_b` scale variants
in favor of two settings that both resample every rep.

BOTH settings use the SAME mean-gap resampling mechanism, generalized from
the earlier design to support any number of generic anchors (was hardcoded
to exactly 3; `amplified_imbalance` needs 8): mirrors Experiment 3's own
exp3_3 mechanism (bounded mean-gap eligibility, uniform pick among eligible,
not MMD^2 cherry-picking) -- see `_choose_resampled_prototypes`.

MEMORY GUARD OVERRIDE: this script runs at n_jobs=5 and passes
`memory_guard_n_workers`/`memory_guard_safety_factor` (from
configs/exp4.yaml's own `memory_guard` block) through to every `kabc.fit`
call via `scripts.common._kabc_fit`'s passthrough kwargs -- real-verified
(not just the relaxed formula), see configs/exp4.yaml's own module comment
for the exact real measurements (peak RSS, concurrent-worker test,
`vm_stat` swap check). This raises kabc's honest convergence ceiling from
k<=7 to k<=8 for both settings; Experiments 1-3 are entirely unaffected
(they call `_kabc_fit` without these kwargs, keeping the original,
unchanged 11-worker/4x-safety assumption).

PARALLELIZATION: across reps, n_jobs=5 (not the usual 10-11) -- `n_jobs`-
sized batches, each a blocking `Parallel()` call (sandbox forces joblib's
'multiprocessing' backend -- see scripts/common.py's module docstring).
Inside one rep's worker, both settings' 2 algorithm rows run sequentially
(kabc first, since unif-kkm-1x needs that same rep's realized tau_kabc).

METRICS: standard tau/stopped/ari/error, plus the per-pair ARI breakdown
(`ari_close_pair`, `ari_far_pair`) and per-size misclassification rate
(`misclass_rate_large`/`_small`) -- both carried over unchanged from the
earlier design; they were essential to interpreting the K=12/imbalance
result there and are exactly what `amplified_imbalance` is designed to
surface here.

SMOKE_TEST=1: reps capped at 5. Settings are NOT shrunk.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# BLAS THREAD CAP -- must happen before numpy (or anything importing it) is
# first imported in this process. See scripts/common.py's module docstring.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import threadpoolctl
import yaml
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment

# SANDBOX: importing scripts.common forces joblib's 'multiprocessing' backend
# process-wide -- see that module's docstring.
from kabc import metrics
from kabc.algorithms.baselines import fit as unif_kkm_fit
from kabc.environments import MovieLensMultiEnv, _movie_rating_distributions, _prototype_bandwidth, compute_pairwise_mmd2
from scripts.common import SMOKE_TEST, SMOKE_TEST_MAX_REPS, _kabc_fit


def _load_movie_pool(data_dir: Path):
    """Same loader as the earlier design -- movie_ids (fixed row/column
    order), means, titles, pools (movie_id -> ratings array), mmd2 (M,M),
    logged for diagnostics only (role selection below is mean-gap based, not
    MMD^2-based -- point 2 of the original design revision).
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


def _eligible(means: np.ndarray, a_idx: int, exclude: set[int], comparison: str, bound: float) -> np.ndarray:
    gaps = np.abs(means - means[a_idx])
    mask = (gaps <= bound) if comparison == "le" else (gaps >= bound)
    mask[a_idx] = False
    for e in exclude:
        mask[e] = False
    return np.where(mask)[0]


def _choose_resampled_prototypes(
    condition_cfg: dict, rep_seed: int, movie_ids: list[int], means: np.ndarray, titles: dict, pools: dict
) -> tuple[MovieLensMultiEnv, list[str], dict]:
    """Per-rep resampling mechanism, generalized from the earlier design to
    support any number of generic anchors (read from `counts_by_role`'s own
    keys, not hardcoded to 3) -- `amplified_imbalance` needs 8, `baseline`
    needs 3. Mirrors Experiment 3's exp3_3 exactly otherwise: bounded
    mean-gap eligibility, UNIFORM pick among eligible, not MMD^2
    cherry-picking.

    1. Close pair: scan a random movie-index permutation for the first A
       with >=1 eligible B where |mean_A-mean_B| <= close_mean_gap_tol;
       uniform pick among eligible B.
    2. Far pair: scan the same permutation (excluding A, B) for the first C
       with >=1 eligible D where |mean_C-mean_D| >= far_mean_gap_min;
       uniform pick among eligible D.
    3. N generic anchors (N = however many `generic_*` roles
       `counts_by_role` has): uniform random draw from the remaining pool
       (excluding A, B, C, D), unconstrained.
    4. Which of (A,B) gets close_a vs close_b, and which of (C,D) gets
       far_large vs far_small, is itself randomized (a coin flip) so role
       assignment doesn't correlate with catalog scan order.

    Returns (env, roles, pair_info) -- pair_info logs the real movie
    identities/gaps chosen this rep, for diagnostics.
    """
    rng = np.random.default_rng(rep_seed * 100 + 1)
    order = rng.permutation(len(movie_ids))
    tol = condition_cfg["close_mean_gap_tol"]
    min_gap = condition_cfg["far_mean_gap_min"]
    counts_by_role = condition_cfg["counts_by_role"]
    generic_roles = sorted(r for r in counts_by_role if r.startswith("generic"))

    a_idx = None
    for cand in order:
        elig = _eligible(means, cand, set(), "le", tol)
        if len(elig) > 0:
            a_idx = int(cand)
            b_idx = int(elig[rng.integers(len(elig))])
            break
    if a_idx is None:
        raise RuntimeError(f"resampled: no movie has an eligible close-pair partner under tol={tol}")

    c_idx = None
    for cand in order:
        if cand in (a_idx, b_idx):
            continue
        elig = _eligible(means, cand, {a_idx, b_idx}, "ge", min_gap)
        if len(elig) > 0:
            c_idx = int(cand)
            d_idx = int(elig[rng.integers(len(elig))])
            break
    if c_idx is None:
        raise RuntimeError(f"resampled: no movie has an eligible far-pair partner under min_gap={min_gap}")

    if rng.integers(2) == 1:
        a_idx, b_idx = b_idx, a_idx
    if rng.integers(2) == 1:
        c_idx, d_idx = d_idx, c_idx

    used = {a_idx, b_idx, c_idx, d_idx}
    remaining = [i for i in range(len(movie_ids)) if i not in used]
    generic_idx = rng.choice(remaining, size=len(generic_roles), replace=False)

    role_idx = {"close_a": a_idx, "close_b": b_idx, "far_large": c_idx, "far_small": d_idx}
    for gi, role in enumerate(generic_roles):
        role_idx[role] = int(generic_idx[gi])

    roles = list(role_idx.keys())
    ids = [movie_ids[role_idx[r]] for r in roles]
    counts = [counts_by_role[r] for r in roles]
    env = MovieLensMultiEnv([pools[mid] for mid in ids], counts, ids, roles)

    close_gap = float(abs(means[a_idx] - means[b_idx]))
    far_gap = float(abs(means[c_idx] - means[d_idx]))
    pair_info = {
        "close_a_movie_id": movie_ids[a_idx], "close_b_movie_id": movie_ids[b_idx],
        "close_a_title": titles[movie_ids[a_idx]], "close_b_title": titles[movie_ids[b_idx]],
        "close_mean_gap": close_gap,
        "far_large_movie_id": movie_ids[c_idx], "far_small_movie_id": movie_ids[d_idx],
        "far_large_title": titles[movie_ids[c_idx]], "far_small_title": titles[movie_ids[d_idx]],
        "far_mean_gap": far_gap,
    }
    return env, roles, pair_info


def _size_buckets(counts: list[int]) -> list[str]:
    """Median split of this instance's own prototype arm counts -- 'large'
    if >= median, else 'small'. Computed per-instance, since the two
    settings use very different count scales.
    """
    median = float(np.median(counts))
    return ["large" if c >= median else "small" for c in counts]


def _restricted_ari(partition: np.ndarray, true_partition: np.ndarray, arm_mask: np.ndarray) -> float:
    """ARI restricted to just the arms in arm_mask (e.g. one role-pair's own
    arms), ignoring every other arm's label.
    """
    return metrics.ari(partition[arm_mask], true_partition[arm_mask])


def _per_arm_correctness(partition: np.ndarray, true_partition: np.ndarray, K: int) -> np.ndarray:
    """Best-overlap (Hungarian) alignment of recovered clusters to the K true
    clusters, then per-arm correctness under that alignment.
    """
    recovered_labels = np.unique(partition)
    overlap = np.zeros((K, len(recovered_labels)), dtype=int)
    for i, true_k in enumerate(range(K)):
        for j, rec_label in enumerate(recovered_labels):
            overlap[i, j] = int(np.sum((true_partition == true_k) & (partition == rec_label)))
    row_idx, col_idx = linear_sum_assignment(-overlap)
    true_to_recovered = {int(true_k): int(recovered_labels[col]) for true_k, col in zip(row_idx, col_idx)}
    correct = np.array([
        partition[i] == true_to_recovered.get(int(true_partition[i]), -999)
        for i in range(len(partition))
    ])
    return correct


def _fit_one_algorithm(algo_id: str, env, delta: float, condition_cfg: dict, rng: np.random.Generator,
                        tau_kabc: float | None, memory_guard_kwargs: dict):
    if algo_id == "kabc":
        return _kabc_fit(env, delta, condition_cfg["budget_cap"], rng, **memory_guard_kwargs), condition_cfg["budget_cap"]
    if algo_id == "unif-kkm-1x":
        if tau_kabc is None:
            raise RuntimeError("unif-kkm-1x requires kabc's tau to already be known this rep -- kabc must run first")
        this_budget = int(min(tau_kabc, condition_cfg["budget_cap"]))
        return unif_kkm_fit(env, delta, this_budget, rng), this_budget
    raise ValueError(f"unknown algorithm id: {algo_id!r}")


def _process_one_rep_one_setting(
    rep: int, exp_id: int, seed_multiplier: int, setting_name: str, condition_cfg: dict,
    movie_ids, means, titles, pools, delta: float, algo_ids: list[str], setting_idx: int,
    memory_guard_kwargs: dict,
) -> list[dict]:
    rep_seed = seed_multiplier * exp_id + rep
    env, roles, pair_info = _choose_resampled_prototypes(condition_cfg, rep_seed, movie_ids, means, titles, pools)

    N = env.n_arms
    K = env.K
    size_bucket = _size_buckets(env.counts)
    close_mask = np.zeros(N, dtype=bool)
    far_mask = np.zeros(N, dtype=bool)
    large_mask = np.zeros(N, dtype=bool)
    start = 0
    for cluster_idx in range(K):
        c = env.counts[cluster_idx]
        arm_slice = slice(start, start + c)
        if roles[cluster_idx] in ("close_a", "close_b"):
            close_mask[arm_slice] = True
        if roles[cluster_idx] in ("far_large", "far_small"):
            far_mask[arm_slice] = True
        if size_bucket[cluster_idx] == "large":
            large_mask[arm_slice] = True
        start += c

    rows = []
    tau_kabc = None
    for algo_idx, algo_id in enumerate(algo_ids):
        algo_seed = rep_seed * 1000 + 100 + setting_idx * 10 + algo_idx
        rng = np.random.default_rng(algo_seed)
        t0 = time.time()
        with threadpoolctl.threadpool_limits(limits=1):
            result, used_budget = _fit_one_algorithm(algo_id, env, delta, condition_cfg, rng, tau_kabc, memory_guard_kwargs)
        wall_clock_seconds = time.time() - t0
        if algo_id == "kabc":
            tau_kabc = result.tau

        partition = np.asarray(result.partition)
        correct = _per_arm_correctness(partition, env.true_partition, K)
        misclass_rate_large = float(1.0 - correct[large_mask].mean()) if large_mask.any() else float("nan")
        misclass_rate_small = float(1.0 - correct[~large_mask].mean()) if (~large_mask).any() else float("nan")

        row = {
            "exp_id": exp_id, "setting": setting_name, "algorithm": algo_id, "rep": rep,
            "seed": algo_seed, "N": N, "K": K, "budget_cap": used_budget,
            "tau": result.tau, "stopped": result.stopped, "wall_clock_seconds": wall_clock_seconds,
            "partition": partition.tolist(),
            "error": metrics.error(partition, env.true_partition),
            "ari": metrics.ari(partition, env.true_partition),
            "ari_close_pair": _restricted_ari(partition, env.true_partition, close_mask),
            "ari_far_pair": _restricted_ari(partition, env.true_partition, far_mask),
            "misclass_rate_large": misclass_rate_large,
            "misclass_rate_small": misclass_rate_small,
            "n_iterations": len(result.history),
        }
        row.update(pair_info)
        print(f"    rep {rep} {setting_name}/{algo_id}: tau={result.tau} stopped={result.stopped} "
              f"ari={row['ari']:.3f} time={wall_clock_seconds:.1f}s", flush=True)
        rows.append(row)
    return rows


def _process_one_rep(rep, exp_id, seed_multiplier, conditions, movie_ids, means, titles, pools, delta, algo_ids,
                      memory_guard_kwargs):
    all_rows = {}
    for setting_idx, (name, cfg) in enumerate(conditions.items()):
        all_rows[name] = _process_one_rep_one_setting(
            rep, exp_id, seed_multiplier, name, cfg, movie_ids, means, titles, pools, delta, algo_ids, setting_idx,
            memory_guard_kwargs,
        )
    return all_rows


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/exp4.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_id = config["exp_id"]
    delta = config["delta"]
    reps = SMOKE_TEST_MAX_REPS if SMOKE_TEST else config["reps"]
    rep_start = config.get("rep_start", 0)
    seed_multiplier = config["seed_multiplier"]
    data_dir = Path(config["data_dir"])
    n_jobs = config.get("n_jobs", 5)
    algo_ids = config["algorithms"]
    memory_guard_kwargs = {
        "memory_guard_n_workers": config["memory_guard"]["n_workers"],
        "memory_guard_safety_factor": config["memory_guard"]["safety_factor"],
    }

    conditions = config["conditions"]
    if "kabc" in algo_ids:
        algo_ids = ["kabc"] + [a for a in algo_ids if a != "kabc"]

    movie_ids, means, titles, pools, mmd2 = _load_movie_pool(data_dir)

    for name, cfg in conditions.items():
        print(f"exp4: setting '{name}' -- K={len(cfg['counts_by_role'])}, "
              f"N={sum(cfg['counts_by_role'].values())}, budget_cap={cfg['budget_cap']:,}", flush=True)

    rep_indices = list(range(rep_start, reps))
    n_reps_this_run = len(rep_indices)
    print(f"exp4: {n_reps_this_run} reps (rep_start={rep_start}, reps={reps}), "
          f"{len(conditions)} settings, {len(algo_ids)} algorithm rows, n_jobs={n_jobs}, "
          f"memory_guard={memory_guard_kwargs}", flush=True)

    for name in conditions:
        Path(f"results/exp4_{name}").mkdir(parents=True, exist_ok=True)

    all_rows = {}
    for name in conditions:
        existing_path = Path(f"results/exp4_{name}/results.parquet")
        if rep_start > 0 and existing_path.exists():
            existing_df = pd.read_parquet(existing_path)
            all_rows[name] = existing_df.to_dict("records")
            print(f"resume: loaded {len(all_rows[name])} existing rows from {existing_path}", flush=True)
        else:
            all_rows[name] = []

    t_start = time.time()
    n_done = 0
    for batch_start in range(0, len(rep_indices), n_jobs):
        batch = rep_indices[batch_start: batch_start + n_jobs]
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(_process_one_rep)(
                rep, exp_id, seed_multiplier, conditions, movie_ids, means, titles, pools, delta, algo_ids,
                memory_guard_kwargs,
            )
            for rep in batch
        )
        for rows_by_setting in batch_results:
            for name, rows in rows_by_setting.items():
                all_rows[name].extend(rows)
        for name in conditions:
            df = pd.DataFrame(all_rows[name])
            df.to_parquet(f"results/exp4_{name}/results.parquet", index=False)
        n_done += len(batch)
        elapsed = time.time() - t_start
        print(f"{n_done}/{n_reps_this_run} reps done, elapsed={elapsed:.1f}s", flush=True)

    for name in conditions:
        print(f"Wrote {len(all_rows[name])} rows to results/exp4_{name}/results.parquet", flush=True)
    print("EXP4 EXIT CODE: 0")


if __name__ == "__main__":
    main()
