"""Shared utilities for the per-experiment dedicated scripts
(run_experiment1.py, run_experiment2.py, run_experiment3.py).

EXTRACTED FROM run_experiment.py: this repo originally had one shared grid
engine (`run_experiment.py`) that every experiment ran through. Once
Experiment 1's algorithm roster grew real cross-algorithm tau dependencies
(a row needing another row's own realized tau before it can run -- see
run_experiment1.py's module docstring), it moved to its own dedicated
script; Experiment 3 already had one for a different structural reason
(fresh environment per rep). Once Experiment 2 gained the SAME dependency
structure as Experiment 1 (the identical 8-algorithm roster), it moved to
its own dedicated script too -- at which point `run_experiment.py`'s shared
engine had zero remaining users and was deleted outright, per that
decision, rather than left as unused code. This module holds exactly the
pieces multiple dedicated scripts still need (calibrated-fit wrappers, grid
expansion, budget_cap resolution) so none of that logic had to be
duplicated three ways or lost when `run_experiment.py` went away.

NOT included here (deliberately): the old flat `ALGORITHMS`/`ENVIRONMENTS`
registries, `_run_one`/`run_experiment`/`main` (the old engine's own
execution loop, obsolete now that every experiment does reps-outer/
algorithms-sequential scheduling itself), `kabc_oracle_fit` (no longer used
by any current experiment -- Experiment 2's redesign dropped `kabc-oracle`
along with the "price of adaptivity" figure it existed for), and each
script's own family-specific `_smoke_shrink_kwargs` (genuinely different
per family: gaussian_mixture needs K/d-aware logic, the same-mean families
just need N kept even -- not worth forcing into one shared function).

BLAS THREAD CAP AND JOBLIB BACKEND: NOT set here. Each entrypoint script
still sets `OMP_NUM_THREADS`/etc. at its own top, before importing anything
(including this module) -- that has to happen before numpy's first import
ANYWHERE in the process, so it can't be delegated to a module that itself
gets imported after other things. `joblib.parallel_config(backend=
"multiprocessing")` (the sandbox syscall workaround -- see any entrypoint
script's own SANDBOX note for the full detail) IS centralized here, since
it's idempotent and only needs to happen once per process before any
Parallel() call, which importing this module already guarantees.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import joblib
import numpy as np
import yaml

joblib.parallel_config(backend="multiprocessing")

from kabc import metrics
from kabc.algorithms.acb import _fit as _acb_fit_internal
from kabc.algorithms.kabc import _fit as _kabc_fit_internal

CALIBRATION_PATH = Path("configs/calibration.yaml")

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
SMOKE_TEST_MAX_REPS = 5
SMOKE_TEST_MAX_BUDGET_CAP = 10_000  # SMOKE_TEST only -- see _resolve_budget_cap
CAP_SEED_OFFSET = 10_000_000  # clearly out of range of any seed_multiplier*exp_id+rep


def _load_c_cal() -> dict[str, float | None]:
    """Loads configs/calibration.yaml's frozen constants, mapping algorithm
    id -> calibrated const (or None if that algorithm's value is null --
    "not yet calibrated", per that file's own header comment). Missing file
    entirely (e.g. before `make calibrate` has ever run) is treated the same
    as "everything null": every algorithm falls back to its own built-in
    uncalibrated default (const=1.0).
    """
    if not CALIBRATION_PATH.exists():
        print(f"WARNING: {CALIBRATION_PATH} not found -- every algorithm will use its uncalibrated default (const=1.0)")
        return {}
    with open(CALIBRATION_PATH) as f:
        raw = yaml.safe_load(f)["c_cal"]
    c_cal = {k: (v.get("value") if v else None) for k, v in raw.items()}
    # tas-fw and acb-gaussian have no calibratable constant at all -- expected
    # null, not a "forgot to run make calibrate" warning (see their own
    # entries in configs/calibration.yaml). acb-gaussian additionally never
    # even reads _C_CAL (see kabc.algorithms.acb_gaussian's CALIBRATION
    # note) -- it's listed here purely so this loader doesn't warn about it.
    EXPECTED_NULL = {"tas-fw", "acb-gaussian"}
    for algo_id, value in c_cal.items():
        if value is None and algo_id not in EXPECTED_NULL:
            print(f"WARNING: c_cal[{algo_id!r}] is null -- this run will use its uncalibrated default (const=1.0)")
    return c_cal


_C_CAL = _load_c_cal()


def _kabc_fit(env, delta, budget_cap, rng, **memory_guard_kwargs):
    """`**memory_guard_kwargs` passes through to `kabc._fit`'s
    `memory_guard_n_workers`/`memory_guard_safety_factor` kwargs -- empty for
    every existing caller (Experiments 1-3), which keeps the original,
    unchanged 11-worker/4x-safety assumption. Experiment 4's own script is
    the only caller expected to pass these, real-verified for its own
    n_jobs=5 configuration -- see kabc.py's `_fit` docstring and
    docs/experiments.md's Experiment 4 "Memory guard override" note.
    """
    return _kabc_fit_internal(
        env, delta, budget_cap, rng, threshold="variance-aware", const=_C_CAL.get("kabc") or 1.0,
        **memory_guard_kwargs,
    )


def _kabc_fit_2x(env, delta, budget_cap, rng, **memory_guard_kwargs):
    """kabc re-run with its calibrated threshold constant DOUBLED (2 * c_cal) --
    a more conservative (larger) stopping threshold than the calibrated kabc
    row. An ablation over the calibration constant itself (c_cal is ~0.1x the
    theoretical constant, per docs/experiments.md section 0.2), not over
    budget: runs as an independent base row at the grid point's full
    budget_cap, own RNG substream -- unlike the old budget-truncated *-0.5x
    rows this replaces, it is not tied to kabc's own realized tau.
    """
    return _kabc_fit_internal(
        env, delta, budget_cap, rng, threshold="variance-aware",
        const=(_C_CAL.get("kabc") or 1.0) * 2.0, **memory_guard_kwargs,
    )


def _kabc_fit_0p5x(env, delta, budget_cap, rng, **memory_guard_kwargs):
    """kabc re-run with its calibrated threshold constant HALVED (0.5 * c_cal) --
    a less conservative (smaller) stopping threshold. See `_kabc_fit_2x`'s
    docstring for the shared rationale; this is its mirror-image counterpart.
    """
    return _kabc_fit_internal(
        env, delta, budget_cap, rng, threshold="variance-aware",
        const=(_C_CAL.get("kabc") or 1.0) * 0.5, **memory_guard_kwargs,
    )


def _kabc_sg_fit(env, delta, budget_cap, rng):
    return _kabc_fit_internal(env, delta, budget_cap, rng, threshold="subgaussian", const=_C_CAL.get("kabc-sg") or 1.0)


def _acb_fit(env, delta, budget_cap, rng):
    c = _C_CAL.get("acb") or 1.0
    return _acb_fit_internal(env, delta, budget_cap, rng, c1=c, c2=c, c3=c, c4=c, c5=c, c6=c)


def _expand_grid(family_cfg: dict):
    """Yield (swept_param, swept_value, resolved_kwargs) per grid point, one
    knob at a time (docs/experiments.md: "one knob at a time, others at
    base") -- a family with sweep sizes [6,4,3] yields 6+4+3=13 grid points,
    not their Cartesian product. Empty `sweeps` yields exactly one grid
    point (`base` as-is).
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
    """Reduces a family's sweeps to their first/last value (representative
    endpoints) under SMOKE_TEST -- generic across families, since it only
    touches `sweeps`, never family-specific kwargs.
    """
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
    `type: s_star_formula`: `s*^2` is Monte-Carlo-estimated
    (kabc.metrics.estimate_s_star_sq) using the `rng` passed in -- callers
    use a DEDICATED, fixed-seed rng derived from (exp_id, grid point index),
    independent of the experiment's own per-rep seed stream, so estimating
    s* never consumes or perturbs the actual experiment's randomness.

    `budget_multiplier` (default 1.0) scales the resolved budget_cap AFTER
    the formula/fixed value but BEFORE the SMOKE_TEST ceiling below. Neither
    current caller (run_experiment1.py, run_experiment2.py) uses a value
    other than 1.0 -- both experiments' old restricted-budget passes
    (0.25x-of-shared-budget_cap) were dropped in favor of the per-algorithm
    0.5x-of-its-own-realized-tau mechanism, which doesn't touch this
    function at all (it scales an already-realized tau, computed after
    `_resolve_budget_cap` has already run once for the grid point). Kept as
    a parameter rather than removed since it's a real, orthogonal knob this
    function has always exposed, not dead code. `s_star` itself is NEVER
    rescaled by `budget_multiplier`: it is a property of the instance, not
    of how much budget this particular pass chooses to spend.

    Under SMOKE_TEST only, the result is additionally capped at
    SMOKE_TEST_MAX_BUDGET_CAP -- NOT a faithful reduced-scale approximation,
    just a ceiling so a SMOKE_TEST run can never try to allocate a
    multi-GB-to-TB Gram matrix (some grid points, e.g. Experiment 2's
    moment_matched_mixture at low `a`, legitimately compute a huge
    budget_cap via the real formula -- see docs/experiments.md's "Known
    limitation" section). Not applied outside SMOKE_TEST.
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
