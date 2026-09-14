"""CLI: python -m scripts.run_experiment1 [configs/exp1.yaml]

Runs Experiment 1 (docs/experiments.md, "Experiment 1" section): the
Gaussian-mixture Delta/d/N sweeps (13 grid points, one knob at a time),
8 algorithm rows per (grid point, rep). Saves results/exp1/results.parquet.

REDESIGN NOTE: this experiment used to run through `run_experiment.py`'s
shared grid engine (build one environment per grid point, then a single
flat `Parallel` call over every (algorithm, rep) work item). It moved to
this dedicated script once the algorithm roster below grew real cross-
algorithm dependencies within a (grid point, rep) -- several rows need
another row's own realized tau before they can run -- which doesn't fit a
flat, order-independent work-item list. Experiment 2 later gained the
identical 8-algorithm/dependency structure and moved to its own dedicated
script for the same reason, at which point `run_experiment.py` had zero
remaining users and was deleted outright. The pieces every dedicated script
still needs (calibrated-fit wrappers, grid expansion, budget_cap
resolution) now live in `scripts/common.py` -- see that module's docstring.

PARALLELIZATION: reps are the parallelization unit (mirroring
scripts/run_experiment3.py's design), NOT the 8 algorithm rows within one
rep. `rep_indices` (per grid point) are chunked into `n_jobs`-sized batches;
each batch runs one blocking `Parallel(n_jobs=n_jobs)` call over
`_process_one_rep`. INSIDE one call, all 8 algorithm rows run SEQUENTIALLY
in a single worker process -- required by the tau-dependency chains below,
and (as a side benefit, same as Experiment 3) resolves those dependencies
for free with no separate ordering phase. Unlike Experiment 3, the
ENVIRONMENT is built ONCE PER GRID POINT, not once per rep:
`GaussianMixtureEnv`'s cluster means are a deterministic closed-form
function of `(N,K,d,Delta,sigma)` (no randomness at all), so every rep at a
given grid point faces the exact identical instance -- only each
algorithm's own pull-noise rng substream differs by rep.
SANDBOX: importing scripts.common forces joblib's 'multiprocessing' backend
process-wide -- see that module's docstring for the full detail.

ALGORITHM ROSTER (8 rows): kabc, unif-kkm-1x, kabc-0.5x, unif-kkm-0.5x,
kabc-sg, kabc-sg-0.5x, acb, acb-0.5x. Three independent BASE algorithms
(kabc, kabc-sg, acb) are each run once at the grid point's full
`budget_cap`; each is then, separately, re-run at `budget_cap =
min(0.5 * its_own_tau, budget_cap)`, with a fresh independent RNG
substream (not the base run's exact same pulls truncated early) -- testing
how conservative THAT algorithm's own stopping rule is, per rep.

MULTIPLIER CHANGED from an earlier 0.9x to 0.5x (2026-09-14), after directly
inspecting per-rep kabc/kabc-0.9x rows: KABC's doubling-trick schedule
quantizes its own realized tau to a small set of round-boundary checkpoints,
and the FINAL round alone typically costs roughly half of the total realized
tau. A 0.9x budget cut therefore almost always lands short of that final
round regardless of how close to 1.0 the multiplier nominally is -- there is
no way to construct a real "kabc re-run with 10% less budget" against this
schedule; 0.9x and, say, 0.6x test essentially the same thing (does the
PREVIOUS round's partition already resolve correctly), not a fine-grained
margin. 0.5x makes what is actually being tested explicit -- a meaningfully
smaller, still-real budget cut -- rather than implying false precision the
doubling schedule cannot honor.

This replaces the OLD design's separate restricted-budget pass
(`budget_multiplier=0.25`, applied uniformly to every algorithm's shared
budget_cap): that asked "how does a globally-smaller budget degrade every
method," this asks the more specific "how much of THIS method's own
realized budget was margin beyond what a correct answer needed." unif-kkm
is no longer swept over its own independent n-grid (the old design) --
instead, `unif-kkm-1x`/`unif-kkm-0.5x` use `kabc`'s tau specifically (not
kabc-sg's or acb's), at 1.0x/0.5x, testing whether active/adaptive
selection beats naive uniform sampling at a matched budget. See
DERIVED_VARIANTS below for the exact (base algorithm, multiplier, fit
function) mapping, and `_reorder_bases_first` for how the three
independent dependency chains (kabc -> {unif-kkm-1x, kabc-0.5x,
unif-kkm-0.5x}; kabc-sg -> kabc-sg-0.5x; acb -> acb-0.5x) are resolved
regardless of the config file's own algorithm ordering.

BUDGET_CAP: ONE value per grid point (`s_star_formula`, `constant=200`,
s*^2 Monte-Carlo-estimated via `kabc.metrics.estimate_s_star_sq` -- see
`scripts.common._resolve_budget_cap`'s own docstring; NOT a literal
closed-form s*^2, though the formula itself is the paper's exact proven
bound, Theorem 3.1), shared identically across all 8 algorithm rows and
every rep at that grid point. This script always calls it with
`budget_multiplier=1.0` (no `budget_multipliers` sweep dimension anymore,
since the restricted-budget concept it existed for is gone -- see
ALGORITHM ROSTER above).

SMOKE_TEST=1: reps capped, sweeps reduced to first/last value, instance
size capped (`_smoke_shrink_kwargs` below, gaussian_mixture-specific --
scripts.common's `_apply_smoke_test` only handles the generic sweep
reduction; each dedicated script keeps its own family-specific size cap).
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
from kabc.environments import GaussianMixtureEnv
from scripts.common import (
    SMOKE_TEST,
    SMOKE_TEST_MAX_REPS,
    CAP_SEED_OFFSET,
    _acb_fit,
    _apply_smoke_test,
    _expand_grid,
    _kabc_fit,
    _kabc_sg_fit,
    _resolve_budget_cap,
)

SMOKE_TEST_MAX_N = 12  # even, and divisible by K in {2,3,4} -- gaussian_mixture-specific, stays local

BASE_FIT_FUNCTIONS = {
    "kabc": _kabc_fit,
    "kabc-sg": _kabc_sg_fit,
    "acb": _acb_fit,
}
# derived_id -> (base_id whose tau this scales, multiplier, fit function to call)
DERIVED_VARIANTS = {
    "unif-kkm-1x": ("kabc", 1.0, unif_kkm_fit),
    "kabc-0.5x": ("kabc", 0.5, _kabc_fit),
    "unif-kkm-0.5x": ("kabc", 0.5, unif_kkm_fit),
    "kabc-sg-0.5x": ("kabc-sg", 0.5, _kabc_sg_fit),
    "acb-0.5x": ("acb", 0.5, _acb_fit),
}


def _reorder_bases_first(algo_ids: list[str]) -> list[str]:
    """Moves every base algorithm that has at least one derived dependent
    present in `algo_ids` to the front (preserving relative order), so every
    base resolves before ANY row that needs its tau -- regardless of the
    config file's own ordering. Simpler than an exact topological
    interleave (which would preserve more of the original ordering) since
    there are only 3 independent chains here and their relative order
    doesn't matter for correctness, only that each base precedes its own
    dependents.
    """
    needed_bases = {DERIVED_VARIANTS[a][0] for a in algo_ids if a in DERIVED_VARIANTS}
    bases_first = [a for a in algo_ids if a in needed_bases]
    rest = [a for a in algo_ids if a not in needed_bases]
    return bases_first + rest


def _fit_one_algorithm(algo_id: str, env, delta: float, budget_cap: int, rng: np.random.Generator, tau_by_base: dict[str, float]):
    if algo_id in DERIVED_VARIANTS:
        base_id, multiplier, fit_fn = DERIVED_VARIANTS[algo_id]
        if base_id not in tau_by_base:
            raise RuntimeError(
                f"{algo_id} requires {base_id!r}'s tau to already be known for this (grid point, rep) -- "
                f"check algorithm ordering in configs/exp1.yaml"
            )
        this_budget = int(min(multiplier * tau_by_base[base_id], budget_cap))
        return fit_fn(env, delta, this_budget, rng), this_budget
    fit_fn = BASE_FIT_FUNCTIONS[algo_id]
    return fit_fn(env, delta, budget_cap, rng), budget_cap


def _process_one_rep(
    rep: int, exp_id: int, seed_multiplier: int, algo_ids: list[str],
    env, budget_cap: int, delta: float, row_context: dict,
) -> list[dict]:
    """Runs every algorithm row for ONE rep against the grid point's SHARED,
    already-built `env`, sequentially, entirely in-memory. Returns a list of
    result rows (one per algorithm).
    """
    rep_seed = seed_multiplier * exp_id + rep
    tau_by_base: dict[str, float] = {}
    rows = []
    for algo_idx, algo_id in enumerate(algo_ids):
        algo_seed = rep_seed * 100 + algo_idx
        rng = np.random.default_rng(algo_seed)
        t0 = time.time()
        with threadpoolctl.threadpool_limits(limits=1):
            result, used_budget_cap = _fit_one_algorithm(algo_id, env, delta, budget_cap, rng, tau_by_base)
        wall_clock_seconds = time.time() - t0
        if algo_id in BASE_FIT_FUNCTIONS:
            tau_by_base[algo_id] = result.tau

        print(f"    rep {rep} {row_context['label']}/{algo_id}: tau={result.tau} stopped={result.stopped} "
              f"time={wall_clock_seconds:.1f}s", flush=True)

        row = {
            "exp_id": exp_id,
            "family": row_context["family"],
            "swept_param": row_context["swept_param"],
            "swept_value": row_context["swept_value"],
            "algorithm": algo_id,
            "rep": rep,
            "seed": algo_seed,
            "budget_cap": used_budget_cap,
            "s_star": row_context["s_star"],
            "tau": result.tau,
            "stopped": result.stopped,
            "wall_clock_seconds": wall_clock_seconds,
            "partition": result.partition.tolist(),
            "error": metrics.error(result.partition, env.true_partition),
            "ari": metrics.ari(result.partition, env.true_partition),
            "n_iterations": len(result.history),
        }
        for param in ("N", "K", "d", "Delta", "sigma"):
            row[param] = row_context["resolved_kwargs"].get(param)
        rows.append(row)
    return rows


def _smoke_shrink_kwargs(kwargs: dict) -> dict:
    """Caps instance SIZE under SMOKE_TEST -- ported from run_experiment.py's
    now-removed gaussian_mixture branch (SMOKE_TEST's whole point is a fast
    end-to-end sanity check, not a faithful reduced-scale reproduction of
    the real grid).
    """
    kwargs = dict(kwargs)
    K = kwargs.get("K", 3)
    target = max(SMOKE_TEST_MAX_N, 2 * K)
    kwargs["N"] = K * -(-target // K)  # smallest multiple of K >= target
    kwargs["d"] = max(kwargs.get("d", 10), 2 * K)  # GaussianMixtureEnv requires d >= 2K
    return kwargs


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/exp1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_id = config["exp_id"]
    delta = config["delta"]
    reps = SMOKE_TEST_MAX_REPS if SMOKE_TEST else config["reps"]
    rep_start = config.get("rep_start", 0)  # resume/append support: skip [0, rep_start)
                                             # -- e.g. a fast 10-rep pilot first
                                             # (rep_start=0, reps=10), then later a
                                             # follow-up (rep_start=10, reps=50) that
                                             # appends 40 more reps to the SAME file
                                             # instead of recomputing the first 10.
                                             # Assumes the grid/algorithm list is
                                             # unchanged between runs. Default 0
                                             # preserves the original full-range behavior.
    seed_multiplier = config["seed_multiplier"]
    n_jobs = config.get("n_jobs", 10)

    family_cfg = config["instances"][0]
    family = family_cfg["family"]
    if family != "gaussian_mixture":
        raise ValueError(f"scripts.run_experiment1 only supports the gaussian_mixture family, got {family!r}")
    algo_ids = _reorder_bases_first(family_cfg["algorithms"])
    if SMOKE_TEST:
        family_cfg = _apply_smoke_test(family_cfg)
        print(f"SMOKE_TEST=1: reps capped at {SMOKE_TEST_MAX_REPS}, sweeps reduced to endpoints, "
              f"instance size capped (N<={SMOKE_TEST_MAX_N} or equivalent)", flush=True)

    # Grid points: build each environment ONCE here (cheap; env construction
    # is deterministic, no pulls made yet), resolve its shared budget_cap
    # once too -- both reused across every rep/algorithm at that point.
    grid_points = []
    grid_point_index = 0
    for swept_param, swept_value, resolved_kwargs in _expand_grid(family_cfg):
        if SMOKE_TEST:
            resolved_kwargs = _smoke_shrink_kwargs(resolved_kwargs)
        env = GaussianMixtureEnv(**resolved_kwargs)
        cap_rng = np.random.default_rng(CAP_SEED_OFFSET + exp_id * 100_000 + grid_point_index)
        grid_point_index += 1
        budget_cap, s_star = _resolve_budget_cap(family_cfg["budget_cap"], env.n_arms, delta, env, cap_rng, budget_multiplier=1.0)
        grid_points.append(dict(
            swept_param=swept_param, swept_value=swept_value, resolved_kwargs=resolved_kwargs,
            env=env, budget_cap=budget_cap, s_star=s_star,
            label=f"{swept_param}={swept_value}",
        ))

    rep_indices = list(range(rep_start, reps))
    n_reps_this_run = len(rep_indices)
    total_runs = len(grid_points) * n_reps_this_run * len(algo_ids)
    print(f"exp1: {len(grid_points)} grid points, {n_reps_this_run} reps "
          f"(rep_start={rep_start}, reps={reps}), {len(algo_ids)} algorithm rows -- "
          f"{total_runs} (grid point, algorithm, rep) runs total, n_jobs={n_jobs} (parallelizing across reps)", flush=True)

    out_dir = Path("results/exp1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.parquet"

    # Seed all_rows from any EXISTING results (resume/append support):
    # rep_start > 0 means earlier reps were already run and saved in a prior
    # invocation -- load them so the incremental writer below produces a
    # complete file (old reps + new reps), not one that silently drops the
    # old reps. See rep_start's own comment above for the intended workflow.
    if rep_start > 0 and out_path.exists():
        existing_df = pd.read_parquet(out_path)
        all_rows: list[dict] = existing_df.to_dict("records")
        print(f"resume: loaded {len(all_rows)} existing rows from {out_path}", flush=True)
    else:
        all_rows = []
    t_start = time.time()
    for gp_idx, gp in enumerate(grid_points, start=1):
        row_context = dict(family=family, swept_param=gp["swept_param"], swept_value=gp["swept_value"],
                            resolved_kwargs=gp["resolved_kwargs"], s_star=gp["s_star"], label=gp["label"])
        n_done_this_point = 0
        for batch_start in range(0, len(rep_indices), n_jobs):
            batch = rep_indices[batch_start:batch_start + n_jobs]
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(_process_one_rep)(rep, exp_id, seed_multiplier, algo_ids, gp["env"], gp["budget_cap"], delta, row_context)
                for rep in batch
            )
            for rows in batch_results:
                all_rows.extend(rows)
            n_done_this_point += len(batch)

            df = pd.DataFrame(all_rows)
            df.to_parquet(out_path, index=False)
            elapsed = time.time() - t_start
            print(f"exp1: grid point {gp_idx}/{len(grid_points)} ({gp['label']}) -- "
                  f"{n_done_this_point}/{n_reps_this_run} reps done this point, {len(all_rows)} rows so far, "
                  f"elapsed={elapsed:.1f}s", flush=True)

    print(f"Wrote {len(all_rows)} rows to {out_path}")
    print("EXP1 EXIT CODE: 0")


if __name__ == "__main__":
    main()
