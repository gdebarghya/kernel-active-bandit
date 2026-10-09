"""Implements docs/experiments.md section 0.2's calibration protocol:
binary-search the smallest c_cal in [0.01, 1] such that empirical error over
100 calibration runs (20 pull-noise seed groups x 5 reps each, all against
ONE fixed GMM instance -- see the "Calibration instances" correction below)
stays <= delta/2, over 8 bisection iterations -- independently for kabc,
kabc-sg, and acb (section 0.2: "Same protocol applied independently to acb
and tas-fw so no method is disadvantaged"). Invoked by `make calibrate`;
writes configs/calibration.yaml.

DEVIATION FROM THE DOC'S LITERAL PROTOCOL (200 runs / 15 bisection
iterations): reduced to 100 runs and 8 iterations for compute-cost reasons
-- see docs/experiments.md section 0.2 for the recorded reasoning. Runs
within a single empirical-error evaluation are parallelized
(joblib.Parallel, n_jobs=-2, i.e. all-but-one core), each capped to a single
BLAS thread (env vars below + threadpoolctl) to avoid the n_jobs-way
oversubscription that otherwise contends 11 multi-threaded worker processes
against 12 hardware threads (6 physical cores). The search also no longer re-verifies the
theoretical constant (c_cal=1.0) empirically before bisecting -- see
`_binary_search_c_cal`'s docstring: that single check was ~79 of kabc's
~100-minute calibration wall-clock, spent reconfirming a constant already
guaranteed safe by the algorithm's own proof (Eq. (4)).

The search interval `[LO, SEARCH_HI]` went through several stages before
settling at its CURRENT value, `[0.1, 1.0]` (see the `LO, SEARCH_HI =`
assignment below) -- docs/experiments.md section 0.2 records the full
history: it was first capped to `[0.01, 0.6]` (kabc/kabc-sg) or
`[0.01, 0.95]` (acb) purely to dodge a cost cliff near `const=1.0` under a
since-abandoned "generous" calibration budget_cap; once calibration was
corrected to use the REAL, formula-derived budget_cap (matching a real
experiment's own budget), that cliff mostly disappeared (kabc's cost near
1.0 dropped from ~520s/run to ~117s/run) and `SEARCH_HI` was widened back
to `1.0`; `LO` was separately raised from `0.01` to `0.1` for a different,
non-cost reason -- an unstable discontinuity found just below `~0.02` where
a run avoids reporting a wrong partition only by exhausting its budget
first, not genuine threshold resolution. If nothing in the current
`[LO, SEARCH_HI]` passes, that is reported and recorded as `null` in
configs/calibration.yaml with an explanatory comment (distinct from tas-fw's
`null`) -- never silently papered over by falling back to an unverified
expensive candidate. Every bisection iteration also prints its own
candidate/error/time immediately (flushed), not just a final summary.

Calibration instances -- CORRECTION to the doc's "20 GMM instances (Exp-1
family, Delta at mid-grid)" (Delta=3, Exp 1's own `base` value and the middle
of its sweep {1,1.5,2,3,4,6}): `GaussianMixtureEnv.__init__` takes no seed
and no other source of randomness -- its cluster means are a deterministic
closed-form function of (N,K,d,Delta,sigma) (see environments.py), matching
how scripts/run_experiment1.py/configs/exp1.yaml already use it (one fixed
instance per grid point, not a per-rep-randomized one). So the 20 "instance seeds"
below (`SEARCH_SEED_BASE + i for i in range(N_INSTANCES)`) do NOT produce 20
different instances -- they build the SAME instance 20 times; each seed only
selects a different pull-noise RNG stream inside `_run_single`. The doc's
literal "20 GMM instances" premise was never achievable with this
environment class (no instance-level randomization exists to seed), not a
bug introduced by this script. What this script actually runs is 100 flat,
independently-seeded reps against one fixed instance -- correct and
well-defined as an empirical error estimate, just not "20 distinct
instances" the way the doc's prose implies. `configs/calibration.yaml`'s
header comment reflects this correction too.

HELD-OUT VALIDATION (beyond the doc's literal protocol, added on review):
after the binary search converges on a candidate c_cal, it is re-run on a
FRESH block of 20x5=100 reps (disjoint seeds from the search block, same
fixed instance -- see the correction above) and that held-out error is
recorded too -- makes it visible in configs/calibration.yaml whether the
calibrated constant actually generalizes to an independent noise sample,
rather than just fitting the search block's particular draws.

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

tas-fw is NOT calibrated by this script: its constant (a multiplier on
beta(t,delta), Yavas et al. Eq. 66) is calibrated separately by
scripts/calibrate_tas_fw.py -> configs/calibration_tas_fw.yaml;
configs/calibration.yaml deliberately records null for it.

SMOKE_TEST=1 reduces instance/rep/iteration counts for fast dev iteration,
per CLAUDE.md.
"""

from __future__ import annotations

import os
import time

# BLAS THREAD CAP -- must be set before numpy (or anything importing it) is
# first imported in THIS process, so it applies whether the process is the
# main one or a joblib/loky worker (loky uses spawn: each worker re-executes
# this module's imports from scratch, inheriting this process's os.environ
# at spawn time). Without this, each of joblib's n_jobs=-2 (11) worker
# processes independently multi-threads its own Gram-matrix/pairwise-distance
# BLAS calls, oversubscribing the 12 hardware threads (6 physical cores) several times over --
# confirmed empirically: an 11-worker real-protocol calibration run showed
# each worker process burning ~2.7x more CPU-seconds than wall-clock elapsed
# (only possible via BLAS-internal threading contending for the same cores),
# and the batch ran far slower than single-job timings predicted. Covers the
# three common BLAS backends since it wasn't confirmed which one numpy/scipy
# resolved to in this environment.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import threadpoolctl
from joblib import Parallel, delayed

# SANDBOX: joblib's default 'loky' backend calls os.sysconf('SC_SEM_NSEMS_MAX')
# at worker-pool startup to size its semaphore pool -- blocked under this
# repo's dev sandbox (a syscall-level restriction, unrelated to the
# filesystem/network permissions the sandbox otherwise governs). 'multiprocessing'
# (fork-based) doesn't call it and was confirmed to run this repo's scripts
# end-to-end under the sandbox; every Parallel(...) call below has no explicit
# backend=, so this sets the process-wide default once, for all of them.
joblib.parallel_config(backend="multiprocessing")

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

# Deviation from docs/experiments.md section 0.2's literal protocol (200
# runs/15 bisection iterations), documented there: compute constraints (this
# script is single-threaded-per-run -- kabc's own runtime dominates, up to
# ~100s+/run at the real, formula-derived calibration budget_cap) made the
# literal protocol impractical for this illustrative-supplement repo. 100 runs
# (20 pull-noise seed groups x 5 reps each, against the ONE fixed GMM instance
# this environment class actually supports -- see the module docstring's
# "Calibration instances" correction; keeps the doc's own "20" rep-group
# count, not 20 distinct instances) and 8 bisection iterations (halving the [LO,SEARCH_HI] search interval
# eight times resolves c_cal to within (SEARCH_HI-LO)/2**8 -- ~0.0035 at the
# current [0.1,1.0]) are used instead; see docs/experiments.md
# section 0.2 for the reasoning recorded alongside the protocol itself.
N_INSTANCES = 4 if SMOKE_TEST else 20
N_REPS_PER_INSTANCE = 2 if SMOKE_TEST else 5  # 20*5 = 100 total calibration runs (see deviation note above)
MAX_BISECT_ITERS = 5 if SMOKE_TEST else 8

# THEORETICAL_CONST=1.0 is the algorithm's own proven threshold constant
# (Eq. (4)) -- never empirically re-verified (see _binary_search_c_cal's
# docstring), only used as the ultimate fallback if the capped search below
# finds nothing safe. SEARCH_HI = 1.0 (full range), not capped below 1.0
# the way an earlier version of this script did: that earlier 0.6 cap was
# justified by cost measured under the 5x-generous budget_cap (~520s/run
# near 1.0). Under the REAL formula-derived budget_cap (see
# `_real_budget_cap`'s docstring for the protocol-correction reasoning), a
# run near const=1.0 gives up much sooner -- measured ~117s/run for kabc,
# ~15s/run for kabc-sg -- so the cost argument for excluding that region no
# longer holds as strongly, AND under a real (not generous) budget, const=1.0
# is no longer safely assumable to satisfy error<=delta/2 purely because it's
# the theoretical constant: the theorem's guarantee doesn't promise
# convergence within any FIXED budget, only eventual correctness given
# unlimited patience. The search still doesn't pre-verify HI=1.0 upfront
# (see _binary_search_c_cal's docstring) -- but if candidates near it are
# needed, the bisection's own FAIL branch (raise lo) will walk arbitrarily
# close to HI on its own, so nothing is silently skipped.
#
# LO raised from 0.01 to 0.1, NOT for compute-cost reasons (kabc.py's
# MAX_ITERATIONS/memory-guard safeguards already make any const in [0.01,1]
# safe to attempt -- nothing hangs or OOMs anymore) but for a correctness
# reason found while investigating why const=0.0192 took 448.6s and still
# failed to converge: const=0.022 (barely larger) converges trivially in
# 2.1s/1 iteration, while const=0.025 does too -- the region just below
# ~0.02 isn't a smooth continuation of "smaller const, harder problem", it's
# an unstable discontinuity where a run only avoids reporting a wrong
# partition by exhausting budget_cap first (stopped=False), not by the
# threshold genuinely resolving edges. const=0.126 (comfortably above the
# new LO) behaves normally (7.8s, real convergence) -- see
# docs/experiments.md section 0.2 for this evidence recorded alongside the
# protocol. A c_cal the search could otherwise land on near old LO=0.01
# would be numerically "fast to validate" (few pulls before hitting budget)
# but not a genuine reflection of the threshold actually separating clusters
# -- not a constant we want calibration to ever freeze.
THEORETICAL_CONST = 1.0
LO, SEARCH_HI = 0.1, 1.0

SEARCH_SEED_BASE = 0
HELD_OUT_SEED_BASE = 10_000  # disjoint from the search block's seeds

OUTPUT_PATH = "configs/calibration.yaml"


def _make_instance(seed: int) -> tuple[GaussianMixtureEnv, np.random.Generator]:
    env = GaussianMixtureEnv(N=CAL_N, K=CAL_K, d=CAL_D, Delta=CAL_DELTA_SEP)
    return env, np.random.default_rng(seed)


def _real_budget_cap() -> int:
    """PROTOCOL CORRECTION (see docs/experiments.md section 0.2): calibration
    must validate error under the SAME budget regime real experiments will
    actually run at, not an artificially generous one. This is the exact
    formula `scripts/common.py`'s `_resolve_budget_cap` uses for a
    `s_star_formula` grid point (constant=200, s_star_sq via
    `estimate_s_star_sq`) -- NO extra margin.

    This replaces a prior `_generous_budget_cap()` (5x this value), which
    was specifically meant to keep calibration runs from ever hitting
    budget_cap, so error would reflect the threshold's tightness alone, not
    budget exhaustion. That reasoning missed that budget exhaustion IS part
    of what real experiments will experience: kabc-sg calibrated under the
    5x-generous cap measured 2% error, but re-run at exp1's real
    grid-point cap (71,055 for this base instance) showed 14% error --
    entirely from `stopped=False` runs that would have resolved given more
    budget, but don't get that budget in a real run. A calibrated constant
    is only meaningful if it was validated against the budget conditions it
    will actually be deployed under.
    """
    env, rng = _make_instance(seed=SEARCH_SEED_BASE)
    s_star_sq = estimate_s_star_sq(env, rng)
    return int(np.ceil(200.0 * env.n_arms / s_star_sq * np.log(env.n_arms / DELTA)))


def _run_single(fit_fn, instance_seed: int, rep: int, budget_cap: int, fit_kwargs: dict) -> bool:
    """One calibration run; returns True iff it counts as an error (didn't
    stop, or stopped with the wrong partition). Rebuilds its own env from
    `instance_seed` rather than receiving one, since GaussianMixtureEnv's own
    construction is deterministic given (N,K,d,Delta) -- cheap, and avoids
    pickling an env object across the joblib worker boundary.

    Wrapped in threadpool_limits(1): belt-and-suspenders alongside the
    module-level BLAS env vars above -- this applies even if some BLAS
    backend was already initialized (with its own thread count baked in)
    before those env vars could take effect in this process, which the
    env-var approach alone can't guarantee against."""
    env, _ = _make_instance(seed=instance_seed)
    rng = np.random.default_rng(instance_seed * 1_000_000 + rep)
    with threadpoolctl.threadpool_limits(limits=1):
        result = fit_fn(env, DELTA, budget_cap, rng, **fit_kwargs)
    return not result.stopped or compute_error(result.partition, env.true_partition) == 1


def _empirical_error(fit_fn, seeds: list[int], budget_cap: int, reps_per_instance: int, fit_kwargs: dict) -> float:
    """Empirical error over `len(seeds) * reps_per_instance` runs, parallel
    across all of them (`Parallel(n_jobs=-2)`, i.e. all-but-one core) --
    these runs are independent (fixed instance/rep/seed grid), so there's no
    reason to pay for them one at a time on a single core."""
    jobs = [
        delayed(_run_single)(fit_fn, instance_seed, rep, budget_cap, fit_kwargs)
        for instance_seed in seeds
        for rep in range(reps_per_instance)
    ]
    errors = Parallel(n_jobs=-2)(jobs)
    return sum(errors) / len(errors)


def _binary_search_c_cal(
    fit_fn, seeds: list[int], budget_cap: int, reps_per_instance: int, const_kwargs_fn, name: str
) -> tuple[float | None, float | None]:
    """Returns (smallest safe c_cal found in [LO, SEARCH_HI], its empirical
    error), or (None, None) if nothing in that capped interval passed.

    Does NOT re-verify `THEORETICAL_CONST=1.0` empirically before searching:
    it's the algorithm's own proven threshold constant (Eq. (4)), already
    guaranteed safe by construction, not something this search needs to
    rediscover -- measured cost for kabc: that single upfront check was ~79
    of ~100 minutes of the whole calibration.

    The search interval itself is also capped to [LO, SEARCH_HI] (SEARCH_HI
    << 1.0): real per-run timing showed cost is flat/cheap below ~0.5 and
    explodes specifically near 1.0, so a plain [LO, 1.0] bisection could
    repeatedly re-walk back into that expensive region while narrowing in
    from above. If NOTHING in [LO, SEARCH_HI] passes, that is itself
    reported (not silently papered over by falling back to an expensive
    candidate) -- the caller decides what to do (e.g. accept
    THEORETICAL_CONST=1.0 unmeasured-in-this-run, or widen the interval).

    Every iteration prints its own candidate/error/time immediately
    (flushed), not just a final summary -- so progress is visible mid-run
    instead of only after all MAX_BISECT_ITERS + held-out validation finish.
    """
    lo, hi = LO, SEARCH_HI
    best_c_cal, best_error = None, None
    for i in range(1, MAX_BISECT_ITERS + 1):
        mid = (lo + hi) / 2.0
        t0 = time.time()
        mid_error = _empirical_error(fit_fn, seeds, budget_cap, reps_per_instance, const_kwargs_fn(mid))
        dt = time.time() - t0
        passed = mid_error <= DELTA / 2.0
        print(
            f"  [{name}] iter {i}/{MAX_BISECT_ITERS}: const={mid:.4f} error={mid_error:.4f} "
            f"{'PASS' if passed else 'FAIL'} time={dt:.1f}s (search interval now "
            f"[{LO if passed else mid:.4f}, {mid if passed else hi:.4f}])",
            flush=True,
        )
        if passed:
            hi = mid
            best_c_cal, best_error = mid, mid_error
        else:
            lo = mid
    if best_c_cal is None:
        print(
            f"  [{name}] WARNING: no c_cal in [{LO},{SEARCH_HI}] passed the (capped) "
            f"bisection search. NOT silently falling back to the theoretical constant "
            f"{THEORETICAL_CONST} -- that would need its own empirical check (expensive, "
            f"skipped by design) before being trusted as 'calibrated'. Returning (None, "
            f"None); caller must decide (widen SEARCH_HI, or accept {THEORETICAL_CONST} "
            f"unmeasured)."
        )
    return best_c_cal, best_error


def _calibrate(name: str, fit_fn, const_kwargs_fn) -> dict | None:
    print(f"Calibrating {name} ...", flush=True)
    t_start = time.time()
    budget_cap = _real_budget_cap()
    search_seeds = [SEARCH_SEED_BASE + i for i in range(N_INSTANCES)]
    held_out_seeds = [HELD_OUT_SEED_BASE + i for i in range(N_INSTANCES)]

    c_cal, calibration_set_error = _binary_search_c_cal(
        fit_fn, search_seeds, budget_cap, N_REPS_PER_INSTANCE, const_kwargs_fn, name
    )
    if c_cal is None:
        # Nothing in the capped search interval passed -- reported already
        # by _binary_search_c_cal. Skip held-out validation (there is no
        # candidate to validate) rather than silently substituting
        # THEORETICAL_CONST and paying for an unrequested expensive check.
        print(f"  {name}: NO c_cal found in [{LO},{SEARCH_HI}]; total search time={time.time() - t_start:.1f}s")
        return None

    t0 = time.time()
    held_out_error = _empirical_error(
        fit_fn, held_out_seeds, budget_cap, N_REPS_PER_INSTANCE, const_kwargs_fn(c_cal)
    )
    print(f"  [{name}] held-out check: error={held_out_error:.4f} time={time.time() - t0:.1f}s", flush=True)

    print(
        f"  {name}: c_cal={c_cal:.4f}  calibration_set_error={calibration_set_error:.4f}  "
        f"held_out_error={held_out_error:.4f}  (delta/2={DELTA / 2.0:.4f})  "
        f"total_time={time.time() - t_start:.1f}s"
    )
    if held_out_error > DELTA / 2.0:
        print(
            f"  WARNING: {name}'s held-out error ({held_out_error:.4f}) exceeds delta/2 "
            f"even though the calibration-set error didn't -- c_cal may not generalize; "
            f"consider more calibration instances/reps."
        )
    return {"value": c_cal, "calibration_set_error": calibration_set_error, "held_out_error": held_out_error}


def _render_yaml(results: dict) -> str:
    def block(name: str, r: dict | None, extra_comment: str = "", null_reason: str = "") -> str:
        if r is None:
            return f"  {name}:\n    value: null\n    calibration_set_error: null\n    held_out_error: null\n{null_reason or extra_comment}"
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
        "    # This value stays null DELIBERATELY -- tas-fw's own calibrated const\n"
        "    # (added 2026-09-17, see kabc.algorithms.tas_fw._fit's `const` kwarg) now\n"
        "    # DOES exist and IS consumed by tas_fw.py, but is calibrated and frozen\n"
        "    # via a completely SEPARATE script/file/protocol -- scripts/calibrate_tas_fw.py\n"
        "    # -> configs/calibration_tas_fw.yaml, NOT this script/file -- because tas-fw\n"
        "    # needed its own synthetic calibration instance (finite-alphabet only, can't\n"
        "    # use GaussianMixtureEnv) and its own cost-driven protocol (a trajectory-\n"
        "    # recording trick specific to tas-fw's const-independent sampling rule; see\n"
        "    # docs/experiments.md's Experiment 3 tas-fw calibration follow-up section for\n"
        "    # the full writeup). This script/file were deliberately left untouched by that\n"
        "    # work to keep the two calibration protocols fully independent.\n"
    )
    acb_gaussian_comment = (
        "    # NOT a \"not yet calibrated\" placeholder like tas-fw above -- this is a\n"
        "    # deliberate, permanent, by-design absence: kabc.algorithms.acb_gaussian\n"
        "    # (arXiv:2406.11485 Appendix A's Gaussian-specialized, non-adaptive ACB\n"
        "    # variant) has NO leading constant anywhere in its formulas to\n"
        "    # calibrate. Every quantity (n_max via a chi-squared quantile; I/J via a\n"
        "    # normal quantile beta and a Monte-Carlo product-quantile alpha) is an\n"
        "    # EXACT delta-level quantile of the statistic's true finite-sample\n"
        "    # distribution, not a concentration-inequality bound with slack to\n"
        "    # tighten -- see acb_gaussian.py's own CALIBRATION note. `make calibrate`\n"
        "    # writes this exact block itself (scripts/calibrate_constants.py) and\n"
        "    # always will -- this is not a hand-edit exception to the rule above.\n"
    )
    # Distinguishes "the capped [LO,SEARCH_HI] search genuinely found nothing
    # safe" (a real, informative negative result) from tas-fw's "never had a
    # constant to search for in the first place" -- both render as `null`,
    # but for a different reason; see _binary_search_c_cal's docstring.
    search_failed_reason = (
        f"    # Capped bisection search over [{LO},{SEARCH_HI}] found NO c_cal with\n"
        f"    # empirical error <= delta/2 -- NOT simply unrun (see\n"
        f"    # scripts/calibrate_constants.py's _binary_search_c_cal docstring).\n"
        f"    # Options: widen SEARCH_HI and re-run `make calibrate`, or fall back to\n"
        f"    # the theoretical constant 1.0 (still empirically unverified against\n"
        f"    # this repo's held-out protocol if you do).\n"
    )

    return (
        "# Populated by `make calibrate` (scripts/calibrate_constants.py), per the\n"
        "# calibration protocol in docs/experiments.md section 0.2: binary-search the\n"
        "# smallest c_cal in [0.01, 1] such that empirical error over 100 calibration\n"
        "# runs stays <= delta/2, over 8 bisection iterations -- a documented deviation\n"
        "# from the doc's literal 200-run/15-iteration protocol; see docs/experiments.md\n"
        "# section 0.2 for the reasoning.\n"
        "# The 100 runs are 20 pull-noise seed groups x 5 reps each, ALL against ONE\n"
        "# fixed GMM instance (Exp-1 family, Delta=3, disjoint seeds) -- NOT 20 distinct\n"
        "# instances: GaussianMixtureEnv's construction is deterministic (no seed\n"
        "# parameter), so there is only one instance to run against; \"20\" here counts\n"
        "# independent noise-seed groups, correcting an earlier inaccurate \"20 GMM\n"
        "# instances\" description. See scripts/calibrate_constants.py's module\n"
        "# docstring for the full correction.\n"
        "# held_out_error is a validation step beyond the doc's literal protocol:\n"
        "# empirical error of the frozen c_cal on a FRESH block of 100 reps (disjoint\n"
        "# seeds from the search block, same fixed instance), to make generalization\n"
        "# to independent noise draws visible.\n"
        "#\n"
        "# Do NOT hand-edit these values -- regenerate via `make calibrate` instead.\n"
        "#\n"
        "# `null` means \"not yet calibrated\": callers should fall back to each\n"
        "# algorithm's own built-in default (const=1.0 -- the uncalibrated,\n"
        "# theoretical constant), NOT treat null as 0 or skip the multiplication.\n"
        "c_cal:\n"
        f"{block('kabc', results.get('kabc'), null_reason=search_failed_reason)}"
        f"{block('kabc-sg', results.get('kabc-sg'), null_reason=search_failed_reason)}"
        f"{block('acb', results.get('acb'), acb_comment, null_reason=search_failed_reason)}"
        f"{block('tas-fw', None, tas_fw_comment)}"
        f"{block('acb-gaussian', None, acb_gaussian_comment)}"
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
    print("Skipping acb-gaussian: no calibratable constant by design (see acb_gaussian.py's CALIBRATION note).")

    with open(OUTPUT_PATH, "w") as f:
        f.write(_render_yaml(results))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
