"""Implements docs/experiments.md section 0.2's calibration protocol:
binary-search the smallest c_cal in [0.01, 1] such that empirical error over
200 calibration runs (20 GMM instances x 10 reps, disjoint seeds) stays
<= delta/2 -- independently for kabc, kabc-sg, and acb (section 0.2: "Same
protocol applied independently to acb and tas-fw so no method is
disadvantaged"). Invoked by `make calibrate`; writes configs/calibration.yaml.

Calibration instances: "20 GMM instances (Exp-1 family, Delta at mid-grid)"
-- the doc doesn't pin an exact Delta, so this uses Delta=3, Exp 1's own
`base` value and the middle of its sweep {1,1.5,2,3,4,6}.

HELD-OUT VALIDATION (beyond the doc's literal protocol, added on review):
after the binary search converges on a candidate c_cal, it is re-run on a
FRESH set of 20 instances (disjoint seeds from the search set) and that
held-out error is recorded too -- makes it visible in configs/calibration.yaml
whether the calibrated constant actually generalizes, rather than just
fitting the search set.

Reaches into each algorithm's PRIVATE `_fit(..., const=...)` (kabc.py) or
`_fit(..., c1=..., ..., c6=...)` (acb.py) rather than the public `fit`/
`fit_sg`: those public functions correctly don't expose a const-override
parameter (CLAUDE.md's fit interface is fixed at (env, delta, budget_cap,
rng)), but this script IS the intended caller of that internal knob --
legitimate same-package internal tooling, not a public-API workaround.

ACB's six constants (c1..c6; see acb.py's module docstring for why they are
NOT structurally interchangeable) are calibrated as a single SHARED
multiplier applied uniformly to all six during the search -- a practical
simplification of the 1-D "binary-search the smallest c_cal" protocol, not a
claim that all six warrant identical scaling.

tas-fw has no calibratable constant in the current tas_fw.py (beta(t,delta),
Eq. 66, is fully specified with no free leading term) -- calibration is
skipped for it; configs/calibration.yaml records null with an explanatory
comment.

SMOKE_TEST=1 reduces instance/rep/iteration counts for fast dev iteration,
per CLAUDE.md.
"""

from __future__ import annotations

import os

import numpy as np

from kabc.algorithms.acb import _fit as _acb_fit
from kabc.algorithms.kabc import _fit as _kabc_fit
from kabc.environments import GaussianMixtureEnv
from kabc.metrics import error as compute_error
from kabc.metrics import estimate_s_star_sq

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

DELTA = 0.05
if SMOKE_TEST:
    # A single real-protocol run (N=60, Delta=3) costs ~100s (many
    # doubling-trick iterations, each O(N^2)) -- fine once for a real
    # calibration, not for "fast dev iteration". SMOKE_TEST swaps in a much
    # smaller, well-separated instance (same one used elsewhere in this
    # repo's own fast smoke tests) so the search actually runs in seconds;
    # it is NOT a faithful reduced-count reproduction of the real protocol,
    # same as how the real SMOKE_TEST convention already changes the grid
    # itself (docs/experiments.md section 0.4: "grid to 2 points"), not just
    # rep counts.
    CAL_N, CAL_K, CAL_D, CAL_DELTA_SEP = 12, 3, 6, 8.0
else:
    CAL_N, CAL_K, CAL_D, CAL_DELTA_SEP = 60, 3, 10, 3.0  # Exp-1's own base instance

N_INSTANCES = 4 if SMOKE_TEST else 20
N_REPS_PER_INSTANCE = 2 if SMOKE_TEST else 10  # 20*10 = 200 total calibration runs
MAX_BISECT_ITERS = 5 if SMOKE_TEST else 15
LO, HI = 0.01, 1.0

SEARCH_SEED_BASE = 0
HELD_OUT_SEED_BASE = 10_000  # disjoint from the search block's seeds

OUTPUT_PATH = "configs/calibration.yaml"


def _make_instance(seed: int) -> tuple[GaussianMixtureEnv, np.random.Generator]:
    env = GaussianMixtureEnv(N=CAL_N, K=CAL_K, d=CAL_D, Delta=CAL_DELTA_SEP)
    return env, np.random.default_rng(seed)


def _generous_budget_cap() -> int:
    """A fixed budget_cap generous enough that calibration runs essentially
    never hit it -- error should reflect the threshold's own tightness, not
    the algorithm running out of patience. Estimated once from the
    theoretical (const=1.0) s_star_sq-formula budget, with a 5x margin.
    """
    env, rng = _make_instance(seed=SEARCH_SEED_BASE)
    s_star_sq = estimate_s_star_sq(env, rng)
    theoretical = 200.0 * env.n_arms / s_star_sq * np.log(env.n_arms / DELTA)
    return int(np.ceil(5.0 * theoretical))


def _empirical_error(fit_fn, seeds: list[int], budget_cap: int, reps_per_instance: int, fit_kwargs: dict) -> float:
    n_runs = 0
    n_errors = 0
    for instance_seed in seeds:
        env, _ = _make_instance(seed=instance_seed)
        for rep in range(reps_per_instance):
            rng = np.random.default_rng(instance_seed * 1_000_000 + rep)
            result = fit_fn(env, DELTA, budget_cap, rng, **fit_kwargs)
            n_runs += 1
            if not result.stopped or compute_error(result.partition, env.true_partition) == 1:
                n_errors += 1
    return n_errors / n_runs


def _binary_search_c_cal(
    fit_fn, seeds: list[int], budget_cap: int, reps_per_instance: int, const_kwargs_fn
) -> tuple[float, float]:
    """Returns (smallest safe c_cal found, its empirical error on `seeds`)."""
    lo, hi = LO, HI
    hi_error = _empirical_error(fit_fn, seeds, budget_cap, reps_per_instance, const_kwargs_fn(hi))
    if hi_error > DELTA / 2.0:
        print(
            f"  WARNING: even c_cal={hi} (the theoretical constant) has empirical "
            f"error {hi_error:.4f} > delta/2={DELTA / 2.0:.4f}; freezing c_cal=1.0 "
            f"without further search (nothing smaller could do better)."
        )
        return hi, hi_error

    best_c_cal, best_error = hi, hi_error
    for _ in range(MAX_BISECT_ITERS):
        mid = (lo + hi) / 2.0
        mid_error = _empirical_error(fit_fn, seeds, budget_cap, reps_per_instance, const_kwargs_fn(mid))
        if mid_error <= DELTA / 2.0:
            hi = mid
            best_c_cal, best_error = mid, mid_error
        else:
            lo = mid
    return best_c_cal, best_error


def _calibrate(name: str, fit_fn, const_kwargs_fn) -> dict:
    print(f"Calibrating {name} ...")
    budget_cap = _generous_budget_cap()
    search_seeds = [SEARCH_SEED_BASE + i for i in range(N_INSTANCES)]
    held_out_seeds = [HELD_OUT_SEED_BASE + i for i in range(N_INSTANCES)]

    c_cal, calibration_set_error = _binary_search_c_cal(
        fit_fn, search_seeds, budget_cap, N_REPS_PER_INSTANCE, const_kwargs_fn
    )
    held_out_error = _empirical_error(
        fit_fn, held_out_seeds, budget_cap, N_REPS_PER_INSTANCE, const_kwargs_fn(c_cal)
    )

    print(
        f"  {name}: c_cal={c_cal:.4f}  calibration_set_error={calibration_set_error:.4f}  "
        f"held_out_error={held_out_error:.4f}  (delta/2={DELTA / 2.0:.4f})"
    )
    if held_out_error > DELTA / 2.0:
        print(
            f"  WARNING: {name}'s held-out error ({held_out_error:.4f}) exceeds delta/2 "
            f"even though the calibration-set error didn't -- c_cal may not generalize; "
            f"consider more calibration instances/reps."
        )
    return {"value": c_cal, "calibration_set_error": calibration_set_error, "held_out_error": held_out_error}


def _render_yaml(results: dict) -> str:
    def block(name: str, r: dict | None, extra_comment: str = "") -> str:
        if r is None:
            return f"  {name}:\n    value: null\n    calibration_set_error: null\n    held_out_error: null\n{extra_comment}"
        return (
            f"  {name}:\n"
            f"    value: {r['value']:.6f}\n"
            f"    calibration_set_error: {r['calibration_set_error']:.6f}\n"
            f"    held_out_error: {r['held_out_error']:.6f}\n"
            f"{extra_comment}"
        )

    acb_comment = (
        "    # Single shared multiplier across the six structurally-distinct\n"
        "    # c1-c6 constants (see acb.py's module docstring) -- a practical\n"
        "    # simplification, not a claim that all six need identical scaling.\n"
    )
    tas_fw_comment = (
        "    # Placeholder only: tas_fw.py's beta(t,delta) (Eq. 66) is fully\n"
        "    # specified with no free leading constant to calibrate -- nothing in\n"
        "    # tas_fw.py consumes this value today. Kept for a possible future\n"
        "    # modification (e.g. scaling beta(t,delta) down empirically).\n"
    )

    return (
        "# Populated by `make calibrate` (scripts/calibrate_constants.py), per the\n"
        "# calibration protocol in docs/experiments.md section 0.2: binary-search the\n"
        "# smallest c_cal in [0.01, 1] such that empirical error over 200 calibration\n"
        "# runs (20 GMM instances, Exp-1 family, disjoint seeds) stays <= delta/2.\n"
        "# held_out_error is a validation step beyond the doc's literal protocol:\n"
        "# empirical error of the frozen c_cal on a FRESH set of 20 instances with\n"
        "# disjoint seeds from the search set, to make generalization visible.\n"
        "#\n"
        "# Do NOT hand-edit these values -- regenerate via `make calibrate` instead.\n"
        "#\n"
        "# `null` means \"not yet calibrated\": callers should fall back to each\n"
        "# algorithm's own built-in default (const=1.0 -- the uncalibrated,\n"
        "# theoretical constant), NOT treat null as 0 or skip the multiplication.\n"
        "c_cal:\n"
        f"{block('kabc', results.get('kabc'))}"
        f"{block('kabc-sg', results.get('kabc-sg'))}"
        f"{block('acb', results.get('acb'), acb_comment)}"
        f"{block('tas-fw', None, tas_fw_comment)}"
    )


def main() -> None:
    if SMOKE_TEST:
        print(f"SMOKE_TEST=1: {N_INSTANCES} instances x {N_REPS_PER_INSTANCE} reps, {MAX_BISECT_ITERS} bisect iters")

    results = {
        "kabc": _calibrate("kabc", _kabc_fit, lambda c: {"threshold": "variance-aware", "const": c}),
        "kabc-sg": _calibrate("kabc-sg", _kabc_fit, lambda c: {"threshold": "subgaussian", "const": c}),
        "acb": _calibrate(
            "acb", _acb_fit, lambda c: {"c1": c, "c2": c, "c3": c, "c4": c, "c5": c, "c6": c}
        ),
    }
    print("Skipping tas-fw: no calibratable constant in the current tas_fw.py (see module docstring).")

    with open(OUTPUT_PATH, "w") as f:
        f.write(_render_yaml(results))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
