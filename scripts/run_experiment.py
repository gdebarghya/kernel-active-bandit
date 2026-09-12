"""CLI: python -m scripts.run_experiment configs/expN.yaml

Loads an experiment config (see configs/exp{1,2,3}.yaml for the schema),
runs every algorithm listed per instance family over its grid, and saves
results as parquet to results/expN/results.parquet.
docs/experiments.md section 0.4: reps=100 (or the config's reps_overrides,
e.g. 50 for tas-fw), seed = seed_multiplier*exp_id + rep, joblib parallelism
(n_jobs=-2), SMOKE_TEST env var for fast iteration.

GRID ITERATION: each instance family's `sweeps` are expanded ONE KNOB AT A
TIME (docs/experiments.md: "one knob at a time, others at base") -- a family
with sweep sizes [6,4,3] yields 6+4+3=13 grid points, not their Cartesian
product; a family with no sweeps yields exactly one grid point (`base`
itself). Note: since every sweep's own value list includes that family's
`base` value for its swept param (e.g. Exp 1's Delta sweep includes Delta=3,
its own base), the base configuration ends up re-run once per sweep
dimension with identical resolved kwargs and identical seeds -- i.e. some
redundant compute at the base point, not a bug, just an accepted
consequence of expanding sweeps independently rather than deduplicating
resolved-kwargs collisions across sweeps.

BUDGET_CAP is resolved FRESH for every individual grid point (never
inherited from `base` or reused across sweep values, per explicit
instruction), computed ONCE per grid point on the main process (not per
algorithm, not per rep, not per worker) and then shared across that grid
point's (algorithm, rep) runs -- it is a property of the instance, not of an
individual repetition's noise realization. For `type: s_star_formula`,
s* is estimated via kabc.metrics.estimate_s_star_sq using a DEDICATED,
fixed-seed rng derived from (exp_id, grid point index) -- independent of the
experiment's own per-rep seed stream (seed_multiplier*exp_id+rep), so
estimating s* never consumes or perturbs the actual experiment's randomness.

kabc-oracle is special-cased: it needs s_star_sq as an extra argument beyond
the standard fit(env,delta,budget_cap,rng) interface (see kabc.py's
fit_oracle docstring) -- only valid for s_star_formula-budget_cap families
(true of every config that lists it, i.e. Exp 2).

RESULT SCHEMA: long format, one row per (grid point, algorithm, rep). See
the ALL_INSTANCE_PARAMS columns below for the full instance-kwarg set (NaN
for whatever doesn't apply to a given family). Includes the raw recovered
`partition` (list column) so notebooks can recompute any metric later
without rerunning, consistent with "notebooks never recompute silently, only
load" (docs/experiments.md section 0.4). Deliberately EXCLUDES the raw
per-algorithm `history` list: its structure differs per algorithm (kabc's
k/n_k/delta_k/... vs acb's l/p/Delta_p/... vs tas-fw's t/Z_t/beta/...) and
doesn't flatten into stable parquet columns; `n_iterations = len(history)` is
kept as a cheap diagnostic instead.

SMOKE_TEST=1: reps capped at 5 (uniformly, including reps_overrides), each
family's sweeps reduced to their first and last value (representative
endpoints, not an arbitrary subset) plus any un-swept base, AND instance
SIZE itself capped (see _smoke_shrink_kwargs) -- consistent with the same
tradeoff already made in calibrate_constants.py's SMOKE_TEST, since a single
real-scale run (N=60, GaussianMixtureEnv) measured ~99s: grid/rep reduction
alone doesn't fix per-run cost, so SMOKE_TEST is not a faithful reduced-scale
reproduction of the real grid, purely a fast sanity check that the code runs
end-to-end.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from pathlib import Path

# BLAS THREAD CAP -- must be set before numpy (or anything importing it) is
# first imported in THIS process, so it applies whether the process is the
# main one or a joblib/loky worker (loky uses spawn: each worker re-executes
# this module's imports from scratch, inheriting this process's os.environ
# at spawn time). Without this, each of joblib's n_jobs=-2 worker processes
# independently multi-threads its own Gram-matrix/pairwise-distance BLAS
# calls, oversubscribing the physical cores several times over -- the same
# issue found and fixed in scripts/calibrate_constants.py (see that module's
# docstring and docs/experiments.md section 0.2 for the measured evidence:
# one worker process burned ~2.7x more CPU-seconds than wall-clock elapsed).
# This file uses the identical joblib.Parallel(n_jobs=-2) pattern and was
# never protected against the same oversubscription -- earlier exp1 timing
# benchmarks run against this script's _run_one likely understate true
# per-run cost for the same reason, though the 11-way process parallelism
# partially masked it there too. Covers the three common BLAS backends since
# it isn't confirmed which one numpy/scipy resolved to in this environment.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import threadpoolctl
import yaml
from joblib import Parallel, delayed

from kabc import metrics
from kabc.algorithms.acb import _fit as _acb_fit_internal
from kabc.algorithms.baselines import fit as unif_kkm_fit
from kabc.algorithms.kabc import _fit as _kabc_fit_internal
from kabc.algorithms.kabc import fit_oracle as kabc_oracle_fit
from kabc.algorithms.tas_fw import fit as tas_fw_fit
from kabc.environments import (
    GaussianMixtureEnv,
    MomentMatchedMixtureEnv,
    MovieLensEnv,
    RingBlobEnv,
    ScalePairEnv,
)

CALIBRATION_PATH = Path("configs/calibration.yaml")


def _load_c_cal() -> dict[str, float | None]:
    """Loads configs/calibration.yaml's frozen constants, mapping algorithm
    id -> calibrated const (or None if that algorithm's value is null --
    "not yet calibrated", per that file's own header comment). Missing file
    entirely (e.g. before `make calibrate` has ever run) is treated the same
    as "everything null": every algorithm falls back to its own built-in
    uncalibrated default (const=1.0).

    This closes a gap this repo had for a while: kabc.py's/acb.py's PUBLIC
    fit/fit_sg entry points correctly don't take a const override (CLAUDE.md's
    fit interface is fixed at (env, delta, budget_cap, rng)), so nothing
    previously read this file at experiment-run time -- calibrating
    constants had no effect on a real `make exp1/2/3` run until this wiring
    was added. See docs/experiments.md section 0.2.
    """
    if not CALIBRATION_PATH.exists():
        print(f"WARNING: {CALIBRATION_PATH} not found -- every algorithm will use its uncalibrated default (const=1.0)")
        return {}
    with open(CALIBRATION_PATH) as f:
        raw = yaml.safe_load(f)["c_cal"]
    c_cal = {k: (v.get("value") if v else None) for k, v in raw.items()}
    for algo_id, value in c_cal.items():
        if value is None and algo_id != "tas-fw":  # tas-fw has no calibratable constant at all -- expected null
            print(f"WARNING: c_cal[{algo_id!r}] is null -- this run will use its uncalibrated default (const=1.0)")
    return c_cal


_C_CAL = _load_c_cal()


def _kabc_fit(env, delta, budget_cap, rng):
    return _kabc_fit_internal(env, delta, budget_cap, rng, threshold="variance-aware", const=_C_CAL.get("kabc") or 1.0)


def _kabc_sg_fit(env, delta, budget_cap, rng):
    return _kabc_fit_internal(env, delta, budget_cap, rng, threshold="subgaussian", const=_C_CAL.get("kabc-sg") or 1.0)


def _acb_fit(env, delta, budget_cap, rng):
    c = _C_CAL.get("acb") or 1.0
    return _acb_fit_internal(env, delta, budget_cap, rng, c1=c, c2=c, c3=c, c4=c, c5=c, c6=c)


ALGORITHMS = {
    "kabc": _kabc_fit,
    "kabc-sg": _kabc_sg_fit,
    "kabc-oracle": kabc_oracle_fit,  # unaffected: fit_oracle has no const override in its fixed signature (see its docstring) -- out of scope here, oracle isn't used by Exp 1
    "acb": _acb_fit,
    "tas-fw": tas_fw_fit,
    "unif-kkm": unif_kkm_fit,
}

ENVIRONMENTS = {
    "gaussian_mixture": GaussianMixtureEnv,
    "scale_pair": ScalePairEnv,
    "moment_matched_mixture": MomentMatchedMixtureEnv,
    "ring_blob": RingBlobEnv,
    "movielens": MovieLensEnv,
    "movielens_reduced": MovieLensEnv,
}

# Union of every environment's constructor kwargs, for the result schema's
# wide instance-param columns (NaN where inapplicable to a given family).
ALL_INSTANCE_PARAMS = [
    "N", "K", "d", "Delta", "sigma",
    "sigma_B", "a", "ring_radius", "ring_noise",
    "data_dir", "arms_per_prototype", "min_ratings", "mean_gap_tol",
]

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
SMOKE_TEST_MAX_REPS = 5
SMOKE_TEST_MAX_N = 12  # even, and divisible by K in {2,3,4} -- see _smoke_shrink_kwargs
SMOKE_TEST_MAX_BUDGET_CAP = 10_000  # SMOKE_TEST only -- see _resolve_budget_cap
CAP_SEED_OFFSET = 10_000_000  # clearly out of range of any seed_multiplier*exp_id+rep


def _smoke_shrink_kwargs(family: str, kwargs: dict) -> dict:
    """Cap instance SIZE under SMOKE_TEST (not just grid/rep counts), applied
    to every grid point's resolved kwargs after expansion -- so even an
    N-sweep's own large values get capped, not just `base`. A single real-
    scale run (N=60, GaussianMixtureEnv) measured ~99s; SMOKE_TEST's whole
    point is fast sanity-checking that the code runs end-to-end, not a
    faithful reduced-scale reproduction of the real grid -- consistent with
    the same tradeoff already made in calibrate_constants.py's SMOKE_TEST.

    Respects each family's own constructor constraints: GaussianMixtureEnv
    needs N divisible by K and d >= 2K; scale_pair/moment_matched_mixture/
    ring_blob need N even (K=2 fixed internally); MovieLensEnv needs enough
    real candidate movies for its K/mean_gap_tol, so K and arms_per_prototype
    are capped rather than N directly.
    """
    kwargs = dict(kwargs)
    if family == "gaussian_mixture":
        K = kwargs.get("K", 3)
        target = max(SMOKE_TEST_MAX_N, 2 * K)
        kwargs["N"] = K * -(-target // K)  # smallest multiple of K >= target
        kwargs["d"] = max(kwargs.get("d", 10), 2 * K)  # GaussianMixtureEnv requires d >= 2K
    elif family in ("scale_pair", "moment_matched_mixture", "ring_blob"):
        kwargs["N"] = SMOKE_TEST_MAX_N if SMOKE_TEST_MAX_N % 2 == 0 else SMOKE_TEST_MAX_N + 1
    elif family in ("movielens", "movielens_reduced"):
        kwargs["K"] = min(kwargs.get("K", 4), 2)
        kwargs["arms_per_prototype"] = min(kwargs.get("arms_per_prototype", 10), 2)
    return kwargs


def _expand_grid(family_cfg: dict):
    """Yield (swept_param, swept_value, resolved_kwargs) per grid point, one
    knob at a time. Empty `sweeps` yields exactly one point (`base` as-is).
    """
    base = family_cfg["base"]
    sweeps = family_cfg.get("sweeps") or []
    if not sweeps:
        yield None, None, dict(base)
        return
    for sweep in sweeps:
        for value in sweep["values"]:
            kwargs = dict(base)
            kwargs[sweep["param"]] = value
            yield sweep["param"], value, kwargs


def _apply_smoke_test(family_cfg: dict) -> dict:
    cfg = copy.deepcopy(family_cfg)
    reduced_sweeps = []
    for sweep in cfg.get("sweeps") or []:
        values = sweep["values"]
        reduced_sweeps.append({"param": sweep["param"], "values": [values[0], values[-1]] if len(values) > 1 else values})
    cfg["sweeps"] = reduced_sweeps
    return cfg


def _resolve_budget_cap(
    budget_cap_cfg: dict, N: int, delta: float, env, rng: np.random.Generator, budget_multiplier: float = 1.0
):
    """Returns (budget_cap: int, s_star: float | None), computed fresh for
    THIS grid point's env/N. `type: fixed` needs no s* (returns None for it).

    `budget_multiplier` (default 1.0, backward compatible) scales the
    resolved budget_cap AFTER the formula/fixed value but BEFORE the
    SMOKE_TEST ceiling below -- e.g. `budget_multiplier=0.25` for Exp 1's
    restricted-budget figure (docs/experiments.md, "how well do these
    methods degrade when queries are scarce"). `s_star` itself is NOT
    rescaled: it is a property of the instance, not of how much budget this
    particular pass is choosing to spend.

    Under SMOKE_TEST only, the result is additionally capped at
    SMOKE_TEST_MAX_BUDGET_CAP. This is NOT a faithful reduced-scale
    approximation and is not applied outside SMOKE_TEST: some grid points
    (e.g. Exp 2's moment_matched_mixture at a=0.5, where the true MMD gap
    approaches 0) legitimately compute a multi-million budget_cap via the
    real formula, and kabc/kabc-sg/kabc-oracle's Gram-matrix computations
    have no sample-size safeguard (see docs/experiments.md's new "Known
    limitation" section) -- attempting that budget_cap for real would try to
    allocate Gram matrices in the hundreds-of-GB-to-TB range well before
    converging. Capping it is only safe for SMOKE_TEST's purpose (verify the
    pipeline's plumbing runs end-to-end), not for measuring anything
    scientific -- the real formula and this cap are two different things.
    """
    cfg_type = budget_cap_cfg["type"]
    if cfg_type == "fixed":
        budget_cap, s_star = int(budget_cap_cfg["value"]), None
    elif cfg_type == "s_star_formula":
        s_star_sq = metrics.estimate_s_star_sq(env, rng)
        constant = budget_cap_cfg["constant"]
        cap = constant * N / s_star_sq * math.log(N / delta)
        budget_cap, s_star = int(math.ceil(cap)), float(math.sqrt(s_star_sq))
    else:
        raise ValueError(f"unknown budget_cap type: {cfg_type!r}")

    budget_cap = int(math.ceil(budget_cap * budget_multiplier))

    if SMOKE_TEST and budget_cap > SMOKE_TEST_MAX_BUDGET_CAP:
        budget_cap = SMOKE_TEST_MAX_BUDGET_CAP
    return budget_cap, s_star


def _run_one(
    exp_id: int,
    family: str,
    swept_param: str | None,
    swept_value,
    resolved_kwargs: dict,
    algo_id: str,
    rep: int,
    seed: int,
    env,
    budget_cap: int,
    s_star: float | None,
    delta: float,
    budget_multiplier: float = 1.0,
) -> dict:
    fit_fn = ALGORITHMS[algo_id]
    rng = np.random.default_rng(seed)

    t0 = time.time()
    # threadpool_limits(1): belt-and-suspenders alongside the module-level
    # BLAS env vars above -- applies even if some BLAS backend was already
    # initialized (with its own thread count baked in) before those env vars
    # could take effect in this process, which the env-var approach alone
    # can't guarantee against.
    with threadpoolctl.threadpool_limits(limits=1):
        if algo_id == "kabc-oracle":
            if s_star is None:
                raise ValueError(
                    f"kabc-oracle requires a s_star_formula budget_cap; family {family!r} uses "
                    f"a fixed budget_cap, so no s* was computed"
                )
            result = fit_fn(env, delta, budget_cap, rng, s_star_sq=s_star**2)
        else:
            result = fit_fn(env, delta, budget_cap, rng)
    wall_clock_seconds = time.time() - t0

    row = {
        "exp_id": exp_id,
        "family": family,
        "algorithm": algo_id,
        "rep": rep,
        "seed": seed,
        "swept_param": swept_param,
        "swept_value": swept_value,
        "budget_cap": budget_cap,
        "budget_multiplier": budget_multiplier,
        "s_star": s_star,
        "tau": result.tau,
        "stopped": result.stopped,
        "wall_clock_seconds": wall_clock_seconds,
        "partition": result.partition.tolist(),
        "error": metrics.error(result.partition, env.true_partition),
        "ari": metrics.ari(result.partition, env.true_partition),
        "n_iterations": len(result.history),
    }
    for param in ALL_INSTANCE_PARAMS:
        row[param] = resolved_kwargs.get(param)
    return row


def run_experiment(config_path: str) -> pd.DataFrame:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_id = config["exp_id"]
    delta = config["delta"]
    default_reps = config["reps"]
    reps_overrides = dict(config.get("reps_overrides") or {})
    seed_multiplier = config["seed_multiplier"]

    if SMOKE_TEST:
        default_reps = min(default_reps, SMOKE_TEST_MAX_REPS)
        reps_overrides = {k: min(v, SMOKE_TEST_MAX_REPS) for k, v in reps_overrides.items()}
        print(
            f"SMOKE_TEST=1: reps capped at {SMOKE_TEST_MAX_REPS}, sweeps reduced to endpoints, "
            f"instance size capped (N<={SMOKE_TEST_MAX_N} or equivalent)"
        )

    # Grid points are collected first (cheap: just resolved kwargs/budget_cap,
    # no algorithm runs yet), THEN executed one grid point at a time below --
    # not one giant flattened work_items list run through a single Parallel()
    # call. A single-call design silently held everything in memory and wrote
    # results.parquet exactly once at the very end: for a real (non-smoke)
    # Exp 1 run (26 grid points x 4 algorithms x 100 reps = 10,400 runs) that
    # meant multi-hour runs with ZERO progress visibility and no partial
    # output to inspect if something needed investigating mid-run. Per-grid-
    # point execution below writes (overwrites, from the full accumulated
    # row list so far) results.parquet after EVERY grid point, and prints a
    # one-line progress update -- so `results/expN/results.parquet` always
    # reflects real, current progress, and a mid-run check no longer requires
    # killing the process to find out how far along it is.
    grid_points = []
    grid_point_index = 0

    for family_cfg in config["instances"]:
        family = family_cfg["family"]
        if SMOKE_TEST:
            family_cfg = _apply_smoke_test(family_cfg)
        env_cls = ENVIRONMENTS[family]

        # budget_multipliers: crossed with EVERY (swept_param, swept_value)
        # grid point below -- not folded into `sweeps` above, since sweeps
        # are one-knob-at-a-time (varying budget_multiplier there would only
        # touch `base`, not all of Delta/d/N's points). Default [1.0] when a
        # family omits the field, so exp2/exp3 (which don't set it) produce
        # EXACTLY the same grid/indexing/output as before this was added --
        # this is the backward-compatibility contract; see the smoke-test
        # diff check in docs/experiments.md's Exp 1 restricted-budget note.
        budget_multipliers = family_cfg.get("budget_multipliers") or [1.0]

        for swept_param, swept_value, resolved_kwargs in _expand_grid(family_cfg):
            if SMOKE_TEST:
                resolved_kwargs = _smoke_shrink_kwargs(family, resolved_kwargs)
            env = env_cls(**resolved_kwargs)
            N = env.n_arms

            for budget_multiplier in budget_multipliers:
                cap_rng = np.random.default_rng(CAP_SEED_OFFSET + exp_id * 100_000 + grid_point_index)
                grid_point_index += 1
                budget_cap, s_star = _resolve_budget_cap(
                    family_cfg["budget_cap"], N, delta, env, cap_rng, budget_multiplier
                )

                work_items = []
                for algo_entry in family_cfg["algorithms"]:
                    algo_id = algo_entry["id"]
                    reps = reps_overrides.get(algo_id, default_reps)
                    for rep in range(reps):
                        seed = seed_multiplier * exp_id + rep
                        work_items.append(
                            dict(
                                exp_id=exp_id,
                                family=family,
                                swept_param=swept_param,
                                swept_value=swept_value,
                                resolved_kwargs=resolved_kwargs,
                                algo_id=algo_id,
                                rep=rep,
                                seed=seed,
                                env=env,
                                budget_cap=budget_cap,
                                s_star=s_star,
                                delta=delta,
                                budget_multiplier=budget_multiplier,
                            )
                        )
                grid_points.append(
                    dict(
                        family=family, swept_param=swept_param, swept_value=swept_value,
                        budget_multiplier=budget_multiplier, work_items=work_items,
                    )
                )

    total_runs = sum(len(gp["work_items"]) for gp in grid_points)
    print(f"exp{exp_id}: {len(grid_points)} grid points, {total_runs} (grid point, algorithm, rep) runs total")

    out_dir = Path(f"results/exp{exp_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.parquet"

    all_rows: list[dict] = []
    run_start = time.time()
    for i, gp in enumerate(grid_points, start=1):
        rows = Parallel(n_jobs=-2)(delayed(_run_one)(**item) for item in gp["work_items"])
        all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        df.to_parquet(out_path, index=False)

        elapsed = time.time() - run_start
        print(
            f"exp{exp_id}: grid point {i}/{len(grid_points)} done "
            f"(family={gp['family']!r}, swept_param={gp['swept_param']!r}, swept_value={gp['swept_value']!r}, "
            f"budget_multiplier={gp['budget_multiplier']}) -- {len(all_rows)}/{total_runs} rows so far, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    print(f"Wrote {len(all_rows)} rows to {out_path}")
    return pd.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a KABC benchmark experiment from a config YAML.")
    parser.add_argument("config", help="Path to configs/expN.yaml")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
