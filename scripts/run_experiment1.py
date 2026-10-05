"""CLI: python -m scripts.run_experiment1 [configs/exp1.yaml]

Runs Experiment 1 (docs/experiments.md, "Experiment 1" section): the
Gaussian-mixture Delta/d/N sweeps (12 grid points, one knob at a time),
6 algorithm rows per (grid point, rep). Saves results/exp1/results.parquet.

REDESIGN NOTE: this experiment used to run through `run_experiment.py`'s
shared grid engine (build one environment per grid point, then a single
flat `Parallel` call over every (algorithm, rep) work item). It moved to
this dedicated script once an earlier algorithm roster grew real cross-
algorithm dependencies within a (grid point, rep) -- some rows needed
another row's own realized tau before they could run -- which doesn't fit
a flat, order-independent work-item list. Experiment 2 gained the
identical structure and moved to its own dedicated script for the same
reason, at which point `run_experiment.py` had zero remaining users and
was deleted outright. The pieces every dedicated script still needs
(calibrated-fit wrappers, grid expansion, budget_cap resolution) now live
in `scripts/common.py` -- see that module's docstring. The dependency that
originally justified a dedicated script (unif-kkm needing kabc's tau) was
itself removed on 2026-10-03 (see ALGORITHM ROSTER below) -- this script
stayed dedicated anyway rather than reverting to a shared engine that no
longer exists.

PARALLELIZATION: reps are the parallelization unit (mirroring
scripts/run_experiment3.py's design), NOT the 6 algorithm rows within one
rep. `rep_indices` (per grid point) are chunked into `n_jobs`-sized batches;
each batch runs one blocking `Parallel(n_jobs=n_jobs)` call over
`_process_one_rep`. INSIDE one call, all 6 algorithm rows run sequentially
in a single worker process (no longer a correctness requirement -- see
ALGORITHM ROSTER below -- just the simplest way to share one built `env`
and avoid a second level of nested parallelism). The ENVIRONMENT is built
ONCE PER GRID POINT, not once per rep: `GaussianMixtureEnv`'s cluster means
are a deterministic closed-form function of `(N,K,d,Delta,sigma)` (no
randomness at all), so every rep at a given grid point faces the exact
identical instance -- only each algorithm's own pull-noise rng substream
differs by rep.
SANDBOX: importing scripts.common forces joblib's 'multiprocessing' backend
process-wide -- see that module's docstring for the full detail.

ALGORITHM ROSTER (6 rows): kabc, kabc-2x, kabc-0.5x, kabc-sg, acb,
acb-gaussian -- all independent (no row's budget_cap or RNG depends on
another row's realized tau), each run once at the grid point's full
`budget_cap`, own RNG substream.

REVISION (2026-10-03): dropped `unif-kkm` entirely (it was tied to kabc's
own realized tau, testing active/adaptive vs. naive uniform sampling at a
matched budget) -- not informative for this experiment's claims, per
direct review of its results. This also removed the only cross-algorithm
tau dependency in the roster, so the dependency-ordering machinery
(`DERIVED_VARIANTS`, `_reorder_bases_first`) that used to live here is
gone too -- every row is now independent. Also dropped the old
budget-truncated self-comparison rows entirely -- kabc-0.5x/kabc-sg-0.5x/
acb-0.5x (each re-run at 0.5x its own realized tau) and unif-kkm-0.5x.
kabc-sg and acb no longer have ANY variant row. In their place,
`kabc-0.5x`/`kabc-2x` are REPURPOSED as an ablation over kabc's own
calibrated threshold CONSTANT instead of its budget: `kabc-2x` doubles
c_cal (more conservative threshold), `kabc-0.5x` halves it (less
conservative) -- see scripts/common.py's `_kabc_fit_2x`/`_kabc_fit_0p5x`
docstrings. Both are independent base rows at the grid point's full
`budget_cap`, NOT tied to kabc's own realized tau the way the name
`kabc-0.5x` previously implied.

ACB-GAUSSIAN (added 2026-09-15, `kabc.algorithms.acb_gaussian`): the
Gaussian-specialized, non-adaptive ACB variant from the ACB paper's own
Appendix A (arXiv:2406.11485) -- the variant its own authors used for their
Figure 1 experiment, distinct from `acb` (the main-text adaptive ACB* this
script also runs). Experiment 1 ONLY (its formulas require a
GaussianMixtureEnv; it hard-fails on anything else -- see
acb_gaussian.py's `_require_gaussian_env`), wired here as an independent
base row with no constant/budget-variant counterpart of its own -- it has
no calibrated-constant slack for a variant row to meaningfully measure
(see acb_gaussian.py's own "NO -0.5x SELF-COMPARISON ROW" note for why
this isn't an oversight) -- and NO calibrated constant (see
`kabc.algorithms.acb_gaussian`'s CALIBRATION note and
`configs/calibration.yaml`'s own note for this algorithm) -- called
directly via `acb_gaussian.fit`, not through a `scripts.common`
calibration-wrapping layer the way `kabc`/`kabc-sg`/`acb` are, since there
is no `c_cal` to inject. It reads Delta/theta/sigma directly off `env`
(known exactly at Experiment 1's instance-construction time) rather than
estimating them -- see acb_gaussian.py's own docstring for why this is a
legitimate, explicitly-flagged non-oracle-cheat comparison specific to this
experiment, and docs/results_exp1_exp2_exp3.tex's Baselines section for the
paper-facing statement of the same.

HISTORY (prior to the 2026-10-03 REVISION above): budget-truncated
self-comparison rows (each base algorithm re-run at a multiplier of its own
realized tau) used to exist for kabc/kabc-sg/acb, and unif-kkm used to come
in `-1x`/`-0.5x` tiers tied to kabc's tau. The multiplier for those rows was
changed from an earlier 0.9x to 0.5x on 2026-09-14, after directly
inspecting per-rep kabc/kabc-0.9x rows: KABC's doubling-trick schedule
quantizes its own realized tau to a small set of round-boundary checkpoints,
and the FINAL round alone typically costs roughly half of the total realized
tau, so a 0.9x cut almost always landed short of that final round regardless
of how close to 1.0 the multiplier nominally was. That entire mechanism, and
unif-kkm itself, are now GONE, per the REVISION above.

BUDGET_CAP: ONE value per grid point (`s_star_formula`, `constant=200`,
s*^2 Monte-Carlo-estimated via `kabc.metrics.estimate_s_star_sq` -- see
`scripts.common._resolve_budget_cap`'s own docstring; NOT a literal
closed-form s*^2, though the formula itself is the paper's exact proven
bound, Theorem 3.1), shared identically across all 6 algorithm rows and
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
from kabc.algorithms import acb_gaussian
from kabc.environments import GaussianMixtureEnv
from scripts.common import (
    SMOKE_TEST,
    SMOKE_TEST_MAX_REPS,
    CAP_SEED_OFFSET,
    _acb_fit,
    _apply_smoke_test,
    _expand_grid,
    _kabc_fit,
    _kabc_fit_0p5x,
    _kabc_fit_2x,
    _kabc_sg_fit,
    _resolve_budget_cap,
)

SMOKE_TEST_MAX_N = 12  # even, and divisible by K in {2,3,4} -- gaussian_mixture-specific, stays local

FIT_FUNCTIONS = {
    "kabc": _kabc_fit,
    "kabc-2x": _kabc_fit_2x,      # kabc with 2 * c_cal -- see scripts.common's docstring
    "kabc-0.5x": _kabc_fit_0p5x,  # kabc with 0.5 * c_cal
    "kabc-sg": _kabc_sg_fit,
    "acb": _acb_fit,
    "acb-gaussian": acb_gaussian.fit,  # no calibration wrapper -- see module docstring's ACB-GAUSSIAN note
}
# All 6 rows are independent -- no row's budget_cap or RNG depends on
# another row's realized tau (unif-kkm, the one row that used to need
# kabc's tau, was dropped 2026-10-03 -- not informative for this
# experiment's claims). So no dependency-ordering/tau-tracking machinery
# is needed here at all, unlike the pre-2026-10-03 design.


def _process_one_rep(
    rep: int, exp_id: int, seed_multiplier: int, algo_ids: list[str],
    env, budget_cap: int, delta: float, row_context: dict,
) -> list[dict]:
    """Runs every algorithm row for ONE rep against the grid point's SHARED,
    already-built `env`, sequentially, entirely in-memory. Returns a list of
    result rows (one per algorithm). Sequential execution is no longer a
    correctness requirement (no row depends on another's tau) -- kept for
    simplicity, since the per-worker cost is dominated by the slowest
    single algorithm (kabc-sg) regardless of ordering.
    """
    rep_seed = seed_multiplier * exp_id + rep
    rows = []
    for algo_idx, algo_id in enumerate(algo_ids):
        algo_seed = rep_seed * 100 + algo_idx
        rng = np.random.default_rng(algo_seed)
        t0 = time.time()
        with threadpoolctl.threadpool_limits(limits=1):
            result = FIT_FUNCTIONS[algo_id](env, delta, budget_cap, rng)
        wall_clock_seconds = time.time() - t0

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
            "budget_cap": budget_cap,
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
    algo_ids = family_cfg["algorithms"]
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
