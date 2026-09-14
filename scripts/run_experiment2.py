"""CLI: python -m scripts.run_experiment2 [configs/exp2.yaml]

Runs Experiment 2 (docs/experiments.md, "Experiment 2" section): three
same-mean, different-shape instance families (scale_pair, moment_matched_
mixture, ring_blob), 6 grid points total (see GRID TRIM NOTE below), 8
algorithm rows per (grid point, rep). Saves results/exp2/results.parquet.

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
scripts/run_experiment1.py's design), NOT the 8 algorithm rows within one
rep. `rep_indices` (per grid point) are chunked into `n_jobs`-sized batches;
each batch runs one blocking `Parallel(n_jobs=n_jobs)` call over
`_process_one_rep`. INSIDE one call, all 8 algorithm rows run SEQUENTIALLY
in a single worker process -- required by the tau-dependency chains below.
The ENVIRONMENT is built ONCE PER GRID POINT, not once per rep: none of
scale_pair/moment_matched_mixture/ring_blob's constructors take an rng or
seed -- their instance geometry is a deterministic closed-form function of
their kwargs, so every rep at a grid point faces the exact identical
instance; only each algorithm's own pull-noise rng substream differs by rep.
SANDBOX: importing scripts.common forces joblib's 'multiprocessing' backend
process-wide -- see that module's docstring for the full detail.

ALGORITHM ROSTER (8 rows): IDENTICAL to Experiment 1's -- kabc,
unif-kkm-1x, kabc-0.5x, unif-kkm-0.5x, kabc-sg, kabc-sg-0.5x, acb,
acb-0.5x. Three independent BASE algorithms (kabc, kabc-sg, acb) are each
run once at the grid point's full `budget_cap`; each is then, separately,
re-run at `budget_cap = min(0.5 * its_own_tau, budget_cap)`, with a fresh
independent RNG substream, testing how conservative THAT algorithm's own
stopping rule is, per rep. `unif-kkm-1x`/`unif-kkm-0.5x` are tied
specifically to `kabc`'s tau, testing whether active/adaptive selection
beats naive uniform sampling at a matched budget. See DERIVED_VARIANTS
below for the exact mapping, and `_reorder_bases_first` for how the three
independent dependency chains are resolved regardless of the config file's
own algorithm ordering.

MULTIPLIER CHANGED from an earlier 0.9x to 0.5x (2026-09-14) -- identical
reasoning to Experiment 1's own module docstring: KABC's doubling-trick
schedule quantizes its own realized tau to a small set of round-boundary
checkpoints, and the FINAL round alone typically costs roughly half of the
total realized tau, so a 0.9x cut cannot actually test "10% less budget" --
it almost always lands in the same "one checkpoint short" bucket as a much
larger cut would. 0.5x is honest about what is actually being tested: a
meaningfully smaller, still-real budget cut.

`kabc-oracle` is GONE -- the old design's "price of adaptivity"
(tau(kabc)/tau(kabc-oracle)) figure is dropped along with it. It was never
actually computable as designed: kabc-oracle's one-shot required budget
exceeded the shared budget_cap formula at all 21 of the old grid points
(0% convergence, always), so the figure never produced real data in the
first place -- removing it isn't a loss of working signal.

BUDGET_CAP: ONE value per grid point (`s_star_formula`, `constant=200`,
s*^2 Monte-Carlo-estimated via `kabc.metrics.estimate_s_star_sq` -- see
`scripts.common._resolve_budget_cap`'s own docstring), shared identically
across all 8 algorithm rows and every rep at that grid point. This script
always calls it with `budget_multiplier=1.0` (no `budget_multipliers`
sweep dimension anymore -- the old restricted-budget pass, 1.0x/0.25x, is
gone, replaced by the 0.5x self-comparison rows above, exactly like
Experiment 1).

GRID TRIM NOTE (2026-09-14): the grid was trimmed a second time, from 10
points down to 6 fixed, illustrative settings (down from the old design's
21, then 10). See docs/experiments.md's Experiment 2 GRID TRIM NOTE for the
full rationale (mirrors notebooks/02_same_mean_shapes.ipynb's move away
from per-family sweep plots to a fixed 6-setting bar-plot structure).

INSTANCE FAMILIES / GRID (6 points total, down from 10, down from the old
design's 21):
- scale_pair: sigma_B in {2, 3} -- 2 points. sigma_B=1.5 DROPPED in the
  original redesign (least-separated, closest to A's own variance, by far
  the slowest point in the old 4-value sweep); sigma_B=4 DROPPED in the
  later GRID TRIM.
- moment_matched_mixture: a in {0.85, 0.9} -- 2 points. The old design's
  extended 0.25x-only range (a in {0.5,0.6,0.7}) was dropped in the
  original redesign -- it existed purely to get ARI-at-cap signal at grid
  points that hit the Gram-matrix memory guard even at full budget
  (docs/experiments.md's "Known limitation" section); with the
  restricted-budget pass gone entirely, that range no longer served the
  purpose it was added for. a=0.8/a=0.95 DROPPED in the later GRID TRIM
  (a=0.8 was also the single slowest point in the whole grid: acb alone
  ran to tau~8M there without converging).
- ring_blob: ring_radius in {2, 3} -- 2 points, up from a single fixed
  ring_radius=2 config in the original redesign (a genuine sweep, matching
  2a/2b's "one knob controls separation" shape). ring_radius=4 DROPPED in
  the later GRID TRIM.

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
from kabc.algorithms.baselines import fit as unif_kkm_fit
from kabc.environments import MomentMatchedMixtureEnv, RingBlobEnv, ScalePairEnv
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

SMOKE_TEST_MAX_N = 12  # even, and divisible by K=2 -- all 3 families fix K=2 internally

ENVIRONMENTS = {
    "scale_pair": ScalePairEnv,
    "moment_matched_mixture": MomentMatchedMixtureEnv,
    "ring_blob": RingBlobEnv,
}
# Union of every environment's constructor kwargs, for the result schema's
# wide instance-param columns (NaN where inapplicable to a given family).
ALL_INSTANCE_PARAMS = ["N", "sigma_B", "a", "ring_radius", "ring_noise"]

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
    config file's own ordering. Identical to run_experiment1.py's helper of
    the same name (kept local, not shared, since it's a handful of lines
    and both scripts' DERIVED_VARIANTS dicts are otherwise independent).
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
                f"check algorithm ordering in configs/exp2.yaml"
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
        algo_ids = _reorder_bases_first(family_cfg["algorithms"])
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
