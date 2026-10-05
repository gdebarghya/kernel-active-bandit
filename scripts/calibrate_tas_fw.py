"""CLI: python -m scripts.calibrate_tas_fw

Calibrates tas-fw's beta(t,delta) threshold (Eq. 66, tas_fw.py) -- the
constant this repo's other three calibratable algorithms (kabc, kabc-sg,
acb) already get via scripts/calibrate_constants.py's live-rerun bisection
search (docs/experiments.md section 0.2). tas-fw needed a SEPARATE script,
not a fourth call into that one, for two structural reasons:

1. CALIBRATION INSTANCE: kabc/kabc-sg/acb calibrate against Exp1's fixed
   GaussianMixtureEnv -- tas-fw can't (finite-alphabet only, by design; see
   tas_fw.py's own module docstring). There is also no fixed reference
   instance anywhere in Experiment 3 (every rep resamples a fresh real
   movie pair) to reuse instead. So this script introduces its own
   calibration-only environment, `kabc.environments.DiscretePMFPairEnv` --
   synthetic, deterministic probability vectors over the real rating
   support {1,...,5}, NOT real MovieLens data (mirrors GaussianMixtureEnv's
   role for the other three: a separate, controlled instance, distinct from
   the real data the calibrated constant will later be deployed against).
   Three regimes (easy/moderate/hard), not one fixed instance, since
   tas-fw's own difficulty is highly gap-sensitive (confirmed directly: the
   ad hoc low-budget probe this calibration follows up on found some real
   pairs resolve almost instantly while near-duplicate pairs need orders of
   magnitude more budget) -- a constant validated on only one gap size
   risks being wrong for the others. Real KL-gap values for the three
   regimes (kabc.algorithms.tas_fw._kl, computed once and hand-verified
   before freezing these PMFs): easy=1.484, moderate=0.804, hard=0.041 --
   a real ~37x spread, not eyeballed.

2. COST: tas-fw's per-run cost at Exp3's real deployment budget_cap=15000
   is ~400-900s (production results/exp3_*/results.parquet: mean
   wall-clock 409-576s across the three versions) -- 5-10x kabc's own
   per-run cost, and the live-rerun bisection protocol (100 runs/iteration
   x 8 iterations + 100 held-out = up to 900 runs) would cost many hours,
   dwarfing kabc's own ~100-minute calibration wall-clock. This script
   avoids that entirely by exploiting a structural fact specific to tas-fw:
   NEITHER the C-tracking sampling rule (`_frank_wolfe_direction`) NOR Z_t
   itself ever reads `beta`/`const` anywhere in tas_fw.py -- the sampled
   trajectory (which arm gets pulled at every t, hence Z_t(t) and
   sigma_hat(t) at every t) is IDENTICAL regardless of const. So each
   calibration rep needs exactly ONE real run -- `tas_fw._fit(...,
   const=1.0, record_full_trajectory=True)`, which ignores the
   const-scaled stop condition and samples all the way to `budget_cap`,
   logging (t, Z_t, beta_raw, partition) at every step -- after which ANY
   candidate const's counterfactual stopping time and correctness are
   recoverable OFFLINE, for free, by scanning the stored trajectory:
       tau(const) = min{t : Z_t(t) >= const * beta_raw(t)}
   This turns "N_INSTANCES x N_REPS x MAX_BISECT_ITERS live re-runs" into
   "N_INSTANCES x N_REPS live runs, done ONCE" followed by a near-instant
   offline sweep over as fine a const grid as desired -- no need to guess a
   search interval upfront the way scripts/calibrate_constants.py's
   [LO, SEARCH_HI] had to be (see that module's docstring for the
   discovery process it took there). This trick is specific to tas-fw's
   structure -- kabc's OWN threshold gates its round-based sampling
   schedule (a round's connectivity decision determines whether the next
   doubling round even happens), so changing kabc's const DOES change what
   kabc samples next; the same shortcut would not be valid for kabc.

GROUND TRUTH USAGE: `env.true_partition` is read ONLY inside this
offline, one-time calibration script (to check each candidate const's
recovered partition against it) -- exactly like kabc/kabc-sg/acb's own
calibration. It is NEVER read inside tas_fw.py's `_fit`/`fit` themselves;
`_fit(..., record_full_trajectory=True)` stores each step's recovered
partition but does not compare it to anything -- this script does that
comparison, after the fact, outside the algorithm. `DiscretePMFPairEnv`'s
PMFs are also synthetic, never real MovieLens data (see point 1 above) --
so nothing about the real Experiment 3 data this constant will be deployed
against is ever seen during calibration either.

ERROR CRITERION (matches scripts/calibrate_constants.py's `_run_single`
exactly, for consistency): a rep counts as an error at a given const if it
EITHER never reaches tau(const) <= budget_cap (uncertified) OR its
recovered partition at tau(const) is wrong -- not just "was the final
answer right regardless of whether it stopped". Empirical error must stay
<= delta/2 in EVERY regime independently (not just pooled/averaged across
regimes) for a candidate const to pass -- a stricter, more defensible bar
than a single pooled average, since a genuine per-instance guarantee
shouldn't be allowed to average an easy regime's near-zero error against a
hard regime's inflated one.

SMOKE_TEST=1: tiny N_REPS_PER_REGIME and budget_cap, for fast dev
iteration -- NOT a faithful reduced-scale reproduction of the real
protocol, same convention as scripts/calibrate_constants.py's own
SMOKE_TEST.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import threadpoolctl
import yaml
from joblib import Parallel, delayed

joblib.parallel_config(backend="multiprocessing")

from kabc.algorithms.tas_fw import _fit as _tas_fw_fit
from kabc.environments import DiscretePMFPairEnv
from kabc.metrics import error as compute_error

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

DELTA = 0.05
ARMS_PER_PROTOTYPE = 4  # N=8, matching configs/exp3.yaml's real deployment scale
BUDGET_CAP = 200 if SMOKE_TEST else 15000  # matches exp3's real budget_cap (see module docstring point 2)
N_REPS_PER_REGIME = 3 if SMOKE_TEST else 30  # see module docstring: real cost per run dominates, not
                                              # iteration count (that part is now free) -- 30/regime x 3
                                              # regimes = 90 real runs for search, another 90 held-out
HELD_OUT_SEED_OFFSET = 100_000  # disjoint from the search block's seeds

# Real KL-gap-verified regimes (see module docstring point 1) -- mirror-image
# PMFs around the support's center value 3, spanning easy/moderate/hard.
REGIMES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "easy": (np.array([0.50, 0.30, 0.10, 0.05, 0.05]), np.array([0.05, 0.05, 0.10, 0.30, 0.50])),
    "moderate": (np.array([0.35, 0.30, 0.20, 0.10, 0.05]), np.array([0.05, 0.10, 0.20, 0.30, 0.35])),
    "hard": (np.array([0.24, 0.22, 0.20, 0.18, 0.16]), np.array([0.16, 0.18, 0.20, 0.22, 0.24])),
}

# Fine offline sweep -- cheap (no re-simulation), so no need to bisect coarsely
# the way scripts/calibrate_constants.py's live-rerun protocol does.
CONST_GRID = np.geomspace(0.001, 1.0, 200)

OUTPUT_DIR = Path("results/tas_fw_calibration")
TRAJECTORIES_PATH = OUTPUT_DIR / "trajectories.pkl"

# Own, separate file -- NOT configs/calibration.yaml (kabc/kabc-sg/acb's
# frozen constants, untouched by this script) -- keeps this calibration's
# code, protocol, and output fully independent of the existing calibration
# infrastructure.
CALIBRATED_CONST_PATH = Path("configs/calibration_tas_fw.yaml")


def _record_one_trajectory(regime: str, pmf_a: np.ndarray, pmf_b: np.ndarray, seed: int) -> dict:
    """One real tas-fw run per calibration rep -- see module docstring point 2
    for why exactly one run (not one per candidate const) suffices."""
    env = DiscretePMFPairEnv(pmf_a, pmf_b, arms_per_prototype=ARMS_PER_PROTOTYPE)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    with threadpoolctl.threadpool_limits(limits=1):
        result = _tas_fw_fit(env, DELTA, BUDGET_CAP, rng, const=1.0, record_full_trajectory=True)
    elapsed = time.time() - t0
    return {
        "regime": regime,
        "seed": seed,
        "true_partition": env.true_partition,
        "history": result.history,
        "elapsed_s": elapsed,
    }


def _record_block(seed_base: int, label: str) -> list[dict]:
    jobs = [
        delayed(_record_one_trajectory)(regime, pmf_a, pmf_b, seed_base + regime_idx * 10_000 + rep)
        for regime_idx, (regime, (pmf_a, pmf_b)) in enumerate(REGIMES.items())
        for rep in range(N_REPS_PER_REGIME)
    ]
    print(f"{label}: recording {len(jobs)} trajectories ({N_REPS_PER_REGIME}/regime x {len(REGIMES)} "
          f"regimes), budget_cap={BUDGET_CAP} ...", flush=True)
    t0 = time.time()
    trajectories = Parallel(n_jobs=-2)(jobs)
    elapsed = time.time() - t0
    mean_run_s = float(np.mean([t["elapsed_s"] for t in trajectories]))
    print(f"{label}: done, wall_clock={elapsed:.1f}s, mean_per_run={mean_run_s:.1f}s", flush=True)
    return trajectories


def _tau_and_correct(traj: dict, const: float) -> tuple[bool, bool]:
    """Offline replay (module docstring point 2): returns (stopped, correct)
    for this ONE candidate const, from the ALREADY-recorded trajectory --
    no re-simulation. Mirrors scripts/calibrate_constants.py's `_run_single`
    error convention exactly."""
    for step in traj["history"]:
        if step["Z_t"] >= const * step["beta_raw"]:
            correct = bool(np.array_equal(step["partition"], traj["true_partition"]))
            return True, correct
    return False, False


def _empirical_error_by_regime(trajectories: list[dict], const: float) -> dict[str, float]:
    errors_by_regime: dict[str, list[bool]] = {regime: [] for regime in REGIMES}
    for traj in trajectories:
        stopped, correct = _tau_and_correct(traj, const)
        errors_by_regime[traj["regime"]].append(not stopped or not correct)
    return {regime: float(np.mean(errs)) for regime, errs in errors_by_regime.items()}


def _find_smallest_safe_const(trajectories: list[dict]) -> tuple[float | None, dict[str, float] | None]:
    """Smallest const in CONST_GRID (ascending) whose per-regime empirical
    error stays <= delta/2 in EVERY regime (module docstring's stricter,
    worst-case-over-regimes bar) -- scanned ascending so the first pass is
    the smallest safe value; cheap enough (no re-simulation) to just check
    every grid point rather than bisect."""
    for const in CONST_GRID:
        errors = _empirical_error_by_regime(trajectories, float(const))
        if all(e <= DELTA / 2.0 for e in errors.values()):
            return float(const), errors
    return None, None


def main() -> None:
    if SMOKE_TEST:
        print(f"SMOKE_TEST=1: {N_REPS_PER_REGIME} reps/regime, budget_cap={BUDGET_CAP}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    search_trajectories = _record_block(seed_base=0, label="search block")
    held_out_trajectories = _record_block(seed_base=HELD_OUT_SEED_OFFSET, label="held-out block")

    with open(TRAJECTORIES_PATH, "wb") as f:
        pickle.dump({"search": search_trajectories, "held_out": held_out_trajectories}, f)
    print(f"wrote raw trajectories to {TRAJECTORIES_PATH} "
          f"(kept for re-examination without re-running the expensive part)", flush=True)

    const, search_errors = _find_smallest_safe_const(search_trajectories)
    if const is None:
        print(f"NO const in [{CONST_GRID.min()},{CONST_GRID.max()}] passed "
              f"(error <= {DELTA / 2.0}) in every regime. Not writing a calibrated value.")
        return

    held_out_errors = _empirical_error_by_regime(held_out_trajectories, const)
    print(f"const={const:.6f}")
    print(f"  search-set errors by regime: {search_errors}")
    print(f"  held-out errors by regime:   {held_out_errors}")
    generalizes = all(err <= DELTA / 2.0 for err in held_out_errors.values())
    for regime, err in held_out_errors.items():
        if err > DELTA / 2.0:
            print(f"  WARNING: held-out error for regime={regime!r} ({err:.4f}) exceeds delta/2 "
                  f"even though the search-set error didn't -- may not generalize.")

    yaml_out = {
        "const": const,
        "delta": DELTA,
        "budget_cap": BUDGET_CAP,
        "arms_per_prototype": ARMS_PER_PROTOTYPE,
        "n_reps_per_regime": N_REPS_PER_REGIME,
        "regimes": list(REGIMES.keys()),
        "search_errors_by_regime": search_errors,
        "held_out_errors_by_regime": held_out_errors,
        "held_out_generalizes": generalizes,
    }
    with open(CALIBRATED_CONST_PATH, "w") as f:
        f.write(
            "# Written by scripts/calibrate_tas_fw.py -- see that script's module\n"
            "# docstring for the full protocol (synthetic DiscretePMFPairEnv "
            "instances,\n# trajectory-recording, offline-only ground truth). "
            "Separate from\n# configs/calibration.yaml (kabc/kabc-sg/acb), "
            "which this script never touches.\n"
            "# Do NOT hand-edit -- regenerate via `python -m scripts.calibrate_tas_fw`.\n"
        )
        yaml.safe_dump(yaml_out, f, default_flow_style=False, sort_keys=False)
    print(f"wrote calibrated const to {CALIBRATED_CONST_PATH}", flush=True)

    print(f"total_time={time.time() - t_start:.1f}s")
    print("EXIT CODE: 0")


if __name__ == "__main__":
    main()
