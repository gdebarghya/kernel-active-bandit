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

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from kabc import metrics
from kabc.algorithms.acb import fit as acb_fit
from kabc.algorithms.baselines import fit as unif_kkm_fit
from kabc.algorithms.kabc import fit as kabc_fit
from kabc.algorithms.kabc import fit_oracle as kabc_oracle_fit
from kabc.algorithms.kabc import fit_sg as kabc_sg_fit
from kabc.algorithms.tas_fw import fit as tas_fw_fit
from kabc.environments import (
    GaussianMixtureEnv,
    MomentMatchedMixtureEnv,
    MovieLensEnv,
    RingBlobEnv,
    ScalePairEnv,
)

ALGORITHMS = {
    "kabc": kabc_fit,
    "kabc-sg": kabc_sg_fit,
    "kabc-oracle": kabc_oracle_fit,
    "acb": acb_fit,
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


def _resolve_budget_cap(budget_cap_cfg: dict, N: int, delta: float, env, rng: np.random.Generator):
    """Returns (budget_cap: int, s_star: float | None), computed fresh for
    THIS grid point's env/N. `type: fixed` needs no s* (returns None for it).

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
) -> dict:
    fit_fn = ALGORITHMS[algo_id]
    rng = np.random.default_rng(seed)

    t0 = time.time()
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

    work_items = []
    grid_point_index = 0

    for family_cfg in config["instances"]:
        family = family_cfg["family"]
        if SMOKE_TEST:
            family_cfg = _apply_smoke_test(family_cfg)
        env_cls = ENVIRONMENTS[family]

        for swept_param, swept_value, resolved_kwargs in _expand_grid(family_cfg):
            if SMOKE_TEST:
                resolved_kwargs = _smoke_shrink_kwargs(family, resolved_kwargs)
            env = env_cls(**resolved_kwargs)
            N = env.n_arms

            cap_rng = np.random.default_rng(CAP_SEED_OFFSET + exp_id * 100_000 + grid_point_index)
            grid_point_index += 1
            budget_cap, s_star = _resolve_budget_cap(family_cfg["budget_cap"], N, delta, env, cap_rng)

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
                        )
                    )

    print(f"exp{exp_id}: {len(work_items)} (grid point, algorithm, rep) runs to execute")
    rows = Parallel(n_jobs=-2)(delayed(_run_one)(**item) for item in work_items)

    df = pd.DataFrame(rows)
    out_dir = Path(f"results/exp{exp_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a KABC benchmark experiment from a config YAML.")
    parser.add_argument("config", help="Path to configs/expN.yaml")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
