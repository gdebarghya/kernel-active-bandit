"""CLI: python -m scripts.run_experiment2 [configs/exp2.yaml]

Runs Experiment 2 (docs/experiments.md, "Experiment 2" section): three
same-mean, different-shape instance families (scale_pair, moment_matched_
mixture, ring_blob), 3 grid points total -- one representative setting per
family (see GRID TRIM NOTE below), 5 algorithm rows per (grid point, rep).
Saves results/exp2/results.parquet.

REDESIGN NOTE: this experiment used to run through `run_experiment.py`'s
shared grid engine, exactly like Experiment 1 originally did. It moved to
this dedicated script once its algorithm roster was changed to match
Experiment 1's exactly (same 8 rows, same tau-dependency chains) -- at
which point `run_experiment.py` had zero remaining users (Experiment 1 and
3 had already left it) and was deleted outright. The pieces every
dedicated script still needs (calibrated-fit wrappers, grid expansion,
budget_cap resolution) now live in `scripts/common.py` -- see that module's
docstring. This script's structure mirrors `run_experiment1.py` almost
exactly; the only real difference is that Experiment 2 has THREE instance
families (each with its own environment class), not one, so the grid-point
loop iterates over `config["instances"]` the way the old shared engine did,
rather than assuming a single family.

PARALLELIZATION: reps are the parallelization unit (mirroring
scripts/run_experiment1.py's design), NOT the 5 algorithm rows within one
rep. `rep_indices` (per grid point) are chunked into `n_jobs`-sized batches;
each batch runs one blocking `Parallel(n_jobs=n_jobs)` call over
`_process_one_rep`. INSIDE one call, all 5 algorithm rows run sequentially
in a single worker process (no longer a correctness requirement -- every
row is independent, see ALGORITHM ROSTER below -- just the simplest way to
share one built `env`). The ENVIRONMENT is built ONCE PER GRID POINT, not
once per rep: none of scale_pair/moment_matched_mixture/ring_blob's
constructors take an rng or seed -- their instance geometry is a
deterministic closed-form function of their kwargs, so every rep at a grid
point faces the exact identical instance; only each algorithm's own
pull-noise rng substream differs by rep.
SANDBOX: importing scripts.common forces joblib's 'multiprocessing' backend
process-wide -- see that module's docstring for the full detail.

ALGORITHM ROSTER (5 rows): matches Experiment 1's current roster minus
`acb-gaussian` (Gaussian-only, not applicable to these non-Gaussian
families) -- kabc, kabc-2x, kabc-0.5x, kabc-sg, acb. All independent (no
row's budget_cap or RNG depends on another row's realized tau), each run
once at the grid point's full `budget_cap`, own RNG substream.

REVISION (2026-10-03): dropped `unif-kkm` entirely (it was tied to kabc's
own realized tau, testing active/adaptive vs. naive uniform sampling at a
matched budget) -- not informative for this experiment's claims, identical
reasoning to Experiment 1's own REVISION. This also removed the only
cross-algorithm tau dependency in the roster, so the dependency-ordering
machinery (`DERIVED_VARIANTS`, `_reorder_bases_first`) that used to live
here is gone too. Also dropped the old budget-truncated self-comparison
rows entirely -- kabc-0.5x/kabc-sg-0.5x/acb-0.5x (each re-run at 0.5x its
own realized tau) and unif-kkm-0.5x. kabc-sg and acb no longer have ANY
variant row. In their place, `kabc-0.5x`/`kabc-2x` are REPURPOSED as an
ablation over kabc's own calibrated threshold CONSTANT instead of its
budget -- identical change and identical rationale to Experiment 1's own
REVISION note; see scripts/common.py's `_kabc_fit_2x`/`_kabc_fit_0p5x`
docstrings.

`kabc-oracle` is GONE -- the old design's "price of adaptivity"
(tau(kabc)/tau(kabc-oracle)) figure is dropped along with it. It was never
actually computable as designed: kabc-oracle's one-shot required budget
exceeded the shared budget_cap formula at all 21 of the old grid points
(0% convergence, always), so the figure never produced real data in the
first place -- removing it isn't a loss of working signal.

BUDGET_CAP: ONE value per grid point (`s_star_formula`, `constant=200`,
s*^2 Monte-Carlo-estimated via `kabc.metrics.estimate_s_star_sq` -- see
`scripts.common._resolve_budget_cap`'s own docstring), shared identically
across all 5 algorithm rows and every rep at that grid point. This script
always calls it with `budget_multiplier=1.0`.

GRID TRIM NOTE (2026-10-03): the grid was trimmed a third time, from 6
settings down to exactly 3 -- one representative setting per family, due
to space constraints in the paper (one illustrative example per family
instead of two). Per-setting real wall-clock timing from the old
6-setting/50-rep run (results/exp2/results.parquet) informed which of
each family's two settings to keep. See docs/experiments.md's Experiment 2
GRID TRIM NOTE (2026-10-03) for the full timing comparison.

INSTANCE FAMILIES / GRID (3 points total, one per family, down from 6):
- scale_pair: sigma_B = 2 (kept; sigma_B=3 dropped).
- moment_matched_mixture: a = 0.9 (kept; a=0.85 dropped -- it was also the
  single most expensive setting in the whole grid, real-measured).
- ring_blob: ring_radius = 2 (kept; ring_radius=3 dropped).

SMOKE_TEST=1: reps capped, sweeps reduced to first/last value, instance
size capped (`_smoke_shrink_kwargs` below -- all three families just need N
kept even, K=2 is fixed internally by each environment class).
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
from kabc.environments import MomentMatchedMixtureEnv, RingBlobEnv, ScalePairEnv
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

SMOKE_TEST_MAX_N = 12  # even, and divisible by K=2 -- all 3 families fix K=2 internally

ENVIRONMENTS = {
    "scale_pair": ScalePairEnv,
    "moment_matched_mixture": MomentMatchedMixtureEnv,
    "ring_blob": RingBlobEnv,
}
# Union of every environment's constructor kwargs, for the result schema's
# wide instance-param columns (NaN where inapplicable to a given family).
ALL_INSTANCE_PARAMS = ["N", "sigma_B", "a", "ring_radius", "ring_noise"]

FIT_FUNCTIONS = {
    "kabc": _kabc_fit,
    "kabc-2x": _kabc_fit_2x,      # kabc with 2 * c_cal -- see scripts.common's docstring
    "kabc-0.5x": _kabc_fit_0p5x,  # kabc with 0.5 * c_cal
    "kabc-sg": _kabc_sg_fit,
    "acb": _acb_fit,
}
# All 5 rows are independent -- no row's budget_cap or RNG depends on
# another row's realized tau (unif-kkm, the one row that used to need
# kabc's tau, was dropped 2026-10-03 -- not informative for this
# experiment's claims, matching Experiment 1's identical change).


def _process_one_rep(
    rep: int, exp_id: int, seed_multiplier: int, algo_ids: list[str],
    env, budget_cap: int, delta: float, row_context: dict,
) -> list[dict]:
    """Runs every algorithm row for ONE rep against the grid point's SHARED,
    already-built `env`, sequentially, entirely in-memory. Returns a list of
    result rows (one per algorithm).
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
        for param in ALL_INSTANCE_PARAMS:
            row[param] = row_context["resolved_kwargs"].get(param)
        rows.append(row)
    return rows


def _smoke_shrink_kwargs(kwargs: dict) -> dict:
    """Caps instance SIZE under SMOKE_TEST -- all three families just need N
    kept even (K=2 fixed internally by each environment class). SMOKE_TEST's
    whole point is a fast end-to-end sanity check, not a faithful reduced-
    scale reproduction of the real grid.
    """
    kwargs = dict(kwargs)
    kwargs["N"] = SMOKE_TEST_MAX_N if SMOKE_TEST_MAX_N % 2 == 0 else SMOKE_TEST_MAX_N + 1
    return kwargs


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/exp2.yaml"
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

    if SMOKE_TEST:
        print(f"SMOKE_TEST=1: reps capped at {SMOKE_TEST_MAX_REPS}, sweeps reduced to endpoints, "
              f"instance size capped (N<={SMOKE_TEST_MAX_N} or equivalent)", flush=True)

    # Grid points: build each environment ONCE here (cheap; env construction
    # is deterministic, no pulls made yet), resolve its shared budget_cap
    # once too -- both reused across every rep/algorithm at that point.
    # Iterates over EVERY family in config["instances"] (unlike Experiment
    # 1, which only ever has one family) -- mirrors the old shared engine's
    # multi-family loop, just without its flat work-item execution.
    grid_points = []
    grid_point_index = 0
    for family_cfg in config["instances"]:
        family = family_cfg["family"]
        env_cls = ENVIRONMENTS[family]
        algo_ids = family_cfg["algorithms"]
        if SMOKE_TEST:
            family_cfg = _apply_smoke_test(family_cfg)

        for swept_param, swept_value, resolved_kwargs in _expand_grid(family_cfg):
            if SMOKE_TEST:
                resolved_kwargs = _smoke_shrink_kwargs(resolved_kwargs)
            env = env_cls(**resolved_kwargs)
            cap_rng = np.random.default_rng(CAP_SEED_OFFSET + exp_id * 100_000 + grid_point_index)
            grid_point_index += 1
            budget_cap, s_star = _resolve_budget_cap(family_cfg["budget_cap"], env.n_arms, delta, env, cap_rng, budget_multiplier=1.0)
            grid_points.append(dict(
                family=family, swept_param=swept_param, swept_value=swept_value, resolved_kwargs=resolved_kwargs,
                env=env, budget_cap=budget_cap, s_star=s_star, algo_ids=algo_ids,
                label=f"{family}/{swept_param}={swept_value}",
            ))

    n_algo_rows = len(grid_points[0]["algo_ids"]) if grid_points else 0
    rep_indices = list(range(rep_start, reps))
    n_reps_this_run = len(rep_indices)
    total_runs = sum(len(gp["algo_ids"]) for gp in grid_points) * n_reps_this_run
    print(f"exp2: {len(grid_points)} grid points, {n_reps_this_run} reps "
          f"(rep_start={rep_start}, reps={reps}), {n_algo_rows} algorithm rows per point -- "
          f"{total_runs} (grid point, algorithm, rep) runs total, n_jobs={n_jobs} (parallelizing across reps)", flush=True)

    out_dir = Path("results/exp2")
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
        row_context = dict(family=gp["family"], swept_param=gp["swept_param"], swept_value=gp["swept_value"],
                            resolved_kwargs=gp["resolved_kwargs"], s_star=gp["s_star"], label=gp["label"])
        n_done_this_point = 0
        for batch_start in range(0, len(rep_indices), n_jobs):
            batch = rep_indices[batch_start:batch_start + n_jobs]
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(_process_one_rep)(rep, exp_id, seed_multiplier, gp["algo_ids"], gp["env"], gp["budget_cap"], delta, row_context)
                for rep in batch
            )
            for rows in batch_results:
                all_rows.extend(rows)
            n_done_this_point += len(batch)

            df = pd.DataFrame(all_rows)
            df.to_parquet(out_path, index=False)
            elapsed = time.time() - t_start
            print(f"exp2: grid point {gp_idx}/{len(grid_points)} ({gp['label']}) -- "
                  f"{n_done_this_point}/{n_reps_this_run} reps done this point, {len(all_rows)} rows so far, "
                  f"elapsed={elapsed:.1f}s", flush=True)

    print(f"Wrote {len(all_rows)} rows to {out_path}")
    print("EXP2 EXIT CODE: 0")


if __name__ == "__main__":
    main()
