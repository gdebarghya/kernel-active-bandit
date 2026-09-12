"""KABC (Algorithm 2, arXiv:2601.07535), its kabc-sg ablation, and kabc-oracle.

kabc/kabc-sg share the same doubling-trick CLUSTER loop (paper's Algorithm 1
composed with Algorithm 2); they differ only in which pairwise threshold
decides same-cluster vs different-cluster edges: variance_aware_threshold
(kabc, Wolfer-Alquier style, Eq. 6) or subgaussian_threshold (kabc-sg,
Tolstikhin style, Discussion section). Since `fit`'s signature is fixed by
CLAUDE.md's shared algorithm interface (env, delta, budget_cap, rng) -- no
room for a threshold-choice parameter without special-casing this algorithm
in experiment scripts -- the choice is made internally: `fit` and `fit_sg`
are both thin wrappers around a shared private `_fit(..., threshold=...)`,
not a subclass hierarchy. There is no per-instance state to justify a class
here, just one branch in the pairwise threshold computation.

kabc-oracle (Discussion section, "Adaptivity") is the SAME CLUSTER subroutine
run exactly once with a fixed oracle budget n* = thresholds.oracle_budget(...)
instead of the doubling-trick schedule, so it shares `_pilot_phase` and
`_cluster_once` with `_fit` rather than duplicating the graph-building logic
in a separate module. Unlike `fit`/`fit_sg`, `fit_oracle` does NOT match
CLAUDE.md's standard 4-arg interface -- see its docstring.

Per-iteration edges: {i,j} is added iff sqrt(mmd2_plugin(Xi,Xj,bandwidth))
<= B^{i,j}(n,delta') (fail to reject same-KME), per Algorithm 1. The
clustering is the connected components of that graph.

n_pilot=5 pilot pulls (median-heuristic bandwidth estimation) count toward
tau, per CLAUDE.md; the bandwidth is then frozen for the rest of the run
(docs/experiments.md section 0.3) and the pilot samples themselves are
discarded, not reused afterward. Each CLUSTER call (whether a doubling-trick
iteration or kabc-oracle's single shot) draws FRESH per-arm samples (matching
Algorithm 1's SAMPLE(n) and the budget bound in Theorem 3.1's proof,
tau <= sum_k N*n_k -- not cumulative reuse across iterations).

K (the true number of clusters) is not a `fit` parameter -- it is read from
`env.K`, since the shared interface has no room for it either; see
environments.py's `Environment` base class.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from kabc.algorithms import Result
from kabc.kernels import median_heuristic_bandwidth, mmd2_plugin, rkhs_variance
from kabc.thresholds import (
    iteration_budget,
    iteration_confidence,
    oracle_budget,
    subgaussian_threshold,
    variance_aware_threshold,
)

N_PILOT = 5

# MAX_ITERATIONS: a generous backstop on the doubling-trick loop's iteration
# count, NOT a memory-safety mechanism (that's MEMORY_GUARD_* below) -- it
# exists only so a future bug (e.g. a broken budget_cap check) can't spin
# forever, and should almost never bind on any legitimate, solvable
# instance. Sizing: n_k = ceil(2**k * log_term) grows geometrically, so even
# this repo's most extreme documented budget_cap (~21M, Exp 2's
# moment_matched_mixture near a=0.5 -- see docs/experiments.md's "Known
# limitation" section) exhausts by k~16-17 via the ordinary budget_cap
# check, and MEMORY_GUARD_* below (independent of N, budget_cap, or delta)
# trips around k~6-10 on any instance whose doubling trick pushes n_k that
# large. MAX_ITERATIONS=50 sits far above where either of those two other
# mechanisms would already have stopped the loop, so it is a true backstop,
# not a third thing that competes with them for which one binds first.
MAX_ITERATIONS = 50

# STOP REASONS -- appended as history[-1] = {"stop_reason": ...} whenever
# the loop/one-shot ends WITHOUT successful convergence (num_clusters == K),
# distinguishing why: ran out of budget mid-schedule (or couldn't afford the
# pilot / next iteration at all), hit the MAX_ITERATIONS backstop, or the
# memory guard refused to attempt an iteration whose Gram matrices would be
# unsafe to allocate. Not a new Result field (Result's schema is shared
# across every algorithm module, per CLAUDE.md's fixed interface -- adding a
# field there needs sign-off beyond this change) -- history is already a
# free-form list[dict], so this reuses it rather than widening Result.
STOP_REASON_BUDGET_CAP = "budget_cap"
STOP_REASON_MAX_ITERATIONS = "max_iterations"
STOP_REASON_MEMORY_GUARD = "memory_guard"

# MEMORY GUARD: _cluster_once's dominant simultaneous allocation is
# mmd2_plugin's three n x n Gram matrices (Kxx, Kyy, Kxy) held alive at once
# within a single pair's call -- NOT all N*(N-1)/2 pairs' matrices at once,
# and not rkhs_variance's (single, smaller) matrix either. Each pair (or
# each arm, for rkhs_variance under the variance-aware threshold) is
# computed and its Gram matrix(es) freed via ordinary CPython refcounting
# before the next one starts -- confirmed by reading _cluster_once itself,
# not assumed. So peak memory scales with n**2 (the per-iteration sample
# size), NOT with N*(N-1)/2 or N (how many such computations happen doesn't
# raise the PEAK, only the total work) -- despite the code path being
# initially motivated by docs/experiments.md's "66 arm-pairs' mmd2_plugin +
# 12 arms' rkhs_variance at N=12" framing, that count describes the number
# of sequential calls, not a simultaneous allocation multiplier.
MEMORY_GUARD_MATRICES_PER_CALL = 3  # Kxx, Kyy, Kxy alive together in mmd2_plugin (the binding case; rkhs_variance's 1 matrix is strictly smaller)
MEMORY_GUARD_SAFETY_FACTOR = 4  # headroom for sklearn/BLAS transient buffers (e.g. rbf_kernel's internal squared-distance intermediates) beyond the 3 persistent arrays, and for imprecise GC timing
MEMORY_GUARD_N_WORKERS = 11  # matches this repo's own joblib Parallel(n_jobs=-2) convention (run_experiment.py, calibrate_constants.py) on a 12-core machine -- every concurrent worker could hit this same peak at once
MEMORY_GUARD_SAFE_TOTAL_BYTES = 8 * 1024**3  # 8 GiB: comfortable headroom under a 16GB target, leaving room for the OS, the Python/numpy baseline footprint, and every other array each worker holds besides the Gram matrices themselves


def fit(env, delta: float, budget_cap: int, rng: np.random.Generator) -> Result:
    """KABC: variance-aware threshold (Wolfer-Alquier style, Eq. 6). Main method.

    K (the number of clusters) is assumed known, per Algorithm 2's KABC(delta, K) --
    it is read from `env.K`, not passed as an argument here (CLAUDE.md's shared
    `fit` interface has no K parameter).
    """
    return _fit(env, delta, budget_cap, rng, threshold="variance-aware")


def fit_sg(env, delta: float, budget_cap: int, rng: np.random.Generator) -> Result:
    """kabc-sg: kernel-uniform sub-Gaussian threshold (Tolstikhin style). Ablation."""
    return _fit(env, delta, budget_cap, rng, threshold="subgaussian")


def fit_oracle(
    env,
    delta: float,
    budget_cap: int,
    rng: np.random.Generator,
    s_star_sq: float,
) -> Result:
    """kabc-oracle: non-adaptive CLUSTER(n*, delta) with oracle n*, per the
    Discussion section's "Adaptivity" paragraph. Used to measure the price of
    adaptivity: tau(kabc) / tau(kabc-oracle).

    Unlike `fit`/`fit_sg`, this does NOT match CLAUDE.md's standard
    `fit(env, delta, budget_cap, rng)` interface: it additionally requires
    `s_star_sq`, the instance's TRUE signal-to-noise ratio (Eq. 7), as an
    oracle input. The paper itself frames the non-adaptive variant this way
    ("takes s* as an input parameter") -- it is not adaptive, so it cannot
    discover s* from its own budgeted samples the way kabc does, and so isn't
    a drop-in for a generic algorithm-registry sweep the way fit/fit_sg are.
    Whoever calls this (the Exp 2 "price of adaptivity" script) must compute
    s_star_sq for the specific instance separately -- that computation is out
    of scope here, exactly as thresholds.py's oracle_budget docstring already
    says of s_star_sq.

    Runs the same pilot phase as fit/fit_sg (counted in tau), then draws
    exactly n* = thresholds.oracle_budget(s_star_sq, N, delta) fresh samples
    per arm in one shot and makes a single clustering decision -- no doubling-
    trick iterations. Always uses the variance-aware threshold (Eq. 6): the
    oracle variant in the paper's Discussion is defined against CLUSTER as
    specified in Algorithm 1, which uses Eq. 6; there is no oracle-sg analogue
    described in docs/experiments.md.

    stopped reflects num_clusters == K on this single attempt. Lemma A.1
    guarantees CLUSTER(n*, delta) is correct with probability >= 1 - delta,
    so across many repetitions at a given delta, stopped should be False on
    roughly a delta (or smaller) fraction of runs -- e.g. at delta=0.05,
    expect stopped=False on the order of 5% of the time or less, not 0%.
    """
    N = env.n_arms
    K = env.K

    tau, bandwidth, cap_result = _pilot_or_graceful_stop(env, N, delta, budget_cap, rng)
    if cap_result is not None:
        return cap_result

    n_star = oracle_budget(s_star_sq, N, delta)
    cost = N * n_star
    if tau + cost > budget_cap:
        warnings.warn(
            f"budget_cap={budget_cap} cannot afford kabc-oracle's one-shot budget "
            f"(pilot={tau} + N*n*={cost} = {tau + cost}); returning stopped=False "
            f"with only the pilot phase completed",
            stacklevel=2,
        )
        return Result(
            partition=np.zeros(N, dtype=int),
            tau=tau,
            stopped=False,
            history=[{"stop_reason": STOP_REASON_BUDGET_CAP}],
        )

    try:
        partition, num_clusters = _cluster_once(
            env, N, n_star, delta, bandwidth, rng, threshold="variance-aware"
        )
    except _MemoryGuardExceeded as e:
        warnings.warn(
            f"memory guard refused kabc-oracle's one-shot n*={n_star} ({e}); "
            f"returning stopped=False with only the pilot phase completed",
            stacklevel=2,
        )
        return Result(
            partition=np.zeros(N, dtype=int),
            tau=tau,
            stopped=False,
            history=[{"stop_reason": STOP_REASON_MEMORY_GUARD, "n_star": n_star}],
        )
    tau += cost
    history = [{"k": 1, "n_k": n_star, "delta_k": delta, "tau": tau, "num_clusters": num_clusters}]

    return Result(partition=partition, tau=tau, stopped=(num_clusters == K), history=history)


def _pull_batch(env, arm: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([np.atleast_1d(np.asarray(env.pull(arm, rng), dtype=float)) for _ in range(n)])


def _pilot_or_graceful_stop(
    env, N: int, delta: float, budget_cap: int, rng: np.random.Generator
) -> tuple[int, float | None, Result | None]:
    """Shared validation + pilot phase for `_fit` and `fit_oracle`.

    Returns (tau, bandwidth, cap_result). If budget_cap can't afford the
    pilot phase, bandwidth is None and cap_result is the graceful Result the
    caller should return immediately; otherwise cap_result is None and the
    caller proceeds with the returned (tau, bandwidth).
    """
    if not 0.0 < delta <= 1.0:
        raise ValueError(f"delta must be in (0, 1], got {delta!r}")
    if budget_cap < 1:
        raise ValueError(f"budget_cap must be >= 1, got {budget_cap!r}")

    pilot_cost = N * N_PILOT
    if budget_cap < pilot_cost:
        # Graceful, not a raise: a Monte Carlo sweep may include a grid point
        # this tight (e.g. an easy instance with a small formula-derived
        # budget_cap), and one bad config shouldn't kill the whole sweep.
        # tau=0 unambiguously marks this case -- every other cap-insufficiency
        # path always has tau >= pilot_cost > 0, since it completes the pilot
        # phase before running out of budget. No pulls are made here, so
        # "pilot pulls count toward tau" (CLAUDE.md) isn't violated: there
        # simply are no pulls to count.
        warnings.warn(
            f"budget_cap={budget_cap} cannot afford the pilot phase (N*n_pilot="
            f"{pilot_cost}, N={N}, n_pilot={N_PILOT}); returning stopped=False "
            f"with no pulls made (tau=0)",
            stacklevel=3,
        )
        return (
            0,
            None,
            Result(
                partition=np.zeros(N, dtype=int),
                tau=0,
                stopped=False,
                history=[{"stop_reason": STOP_REASON_BUDGET_CAP}],
            ),
        )

    # Pilot phase: n_pilot pulls/arm, pooled, for the median-heuristic bandwidth.
    # Counts toward tau; discarded afterward (not reused by the main loop).
    pilot = [_pull_batch(env, i, N_PILOT, rng) for i in range(N)]
    bandwidth = median_heuristic_bandwidth(np.concatenate(pilot, axis=0))
    return pilot_cost, bandwidth, None


class _MemoryGuardExceeded(Exception):
    """Raised by `_check_memory_guard` when the projected Gram-matrix memory
    footprint at a given `n` exceeds MEMORY_GUARD_SAFE_TOTAL_BYTES. Caught by
    `_fit`'s loop and `fit_oracle` to produce a graceful
    Result(stopped=False, ...) instead of risking an OOM or severe thrashing
    slowdown -- see MEMORY_GUARD_* constants' docstrings for the sizing
    reasoning, and docs/experiments.md's "Known limitation" section for the
    real incident (Exp 2's moment_matched_mixture near a=0.5) this guards
    against.
    """

    def __init__(self, n: int, projected_bytes: int):
        self.n = n
        self.projected_bytes = projected_bytes
        super().__init__(
            f"n={n} would need ~{projected_bytes / 1024**3:.2f} GiB across "
            f"{MEMORY_GUARD_N_WORKERS} concurrent workers "
            f"(> {MEMORY_GUARD_SAFE_TOTAL_BYTES / 1024**3:.0f} GiB safety budget)"
        )


def _check_memory_guard(n: int) -> None:
    """Raises _MemoryGuardExceeded if attempting Gram-matrix computations at
    this `n` would be unsafe. Called at the TOP of `_cluster_once`, before
    any samples are pulled or any Gram matrix is built -- if we're going to
    refuse this `n`, there's no reason to spend the pulls first.

    projected_bytes = MEMORY_GUARD_N_WORKERS * MEMORY_GUARD_MATRICES_PER_CALL
        * MEMORY_GUARD_SAFETY_FACTOR * n**2 * 8 (bytes/float64), i.e. every
    concurrent worker independently hitting the same PEAK simultaneous
    allocation (3 Gram matrices of shape (n,n), not N*(N-1)/2 or N of them --
    see MEMORY_GUARD_MATRICES_PER_CALL's docstring for why), times a safety
    margin.
    """
    projected_bytes = (
        MEMORY_GUARD_N_WORKERS * MEMORY_GUARD_MATRICES_PER_CALL * MEMORY_GUARD_SAFETY_FACTOR * (n**2) * 8
    )
    if projected_bytes > MEMORY_GUARD_SAFE_TOTAL_BYTES:
        raise _MemoryGuardExceeded(n, projected_bytes)


def _cluster_once(
    env,
    N: int,
    n: int,
    delta_prime: float,
    bandwidth: float,
    rng: np.random.Generator,
    *,
    threshold: Literal["variance-aware", "subgaussian"],
    const: float = 1.0,
) -> tuple[np.ndarray, int]:
    """One call of Algorithm 1's CLUSTER(n, delta'): draw n fresh samples/arm,
    build the same-cluster graph via the pairwise threshold, return
    (partition, num_clusters) as the graph's connected components.

    `const` multiplies the threshold's entire returned bound (thresholds.py's
    `const` convention, i.e. the calibration protocol's c_cal, section 0.2) --
    defaults to 1.0, the theoretical constant. Not exposed via the public
    `fit`/`fit_sg`; only scripts/calibrate_constants.py is expected to
    override it, via `_fit` below.

    Raises _MemoryGuardExceeded (via `_check_memory_guard`) before pulling
    any samples if this `n` would be unsafe to attempt -- see that function's
    docstring.
    """
    _check_memory_guard(n)
    arm_samples = [_pull_batch(env, i, n, rng) for i in range(N)]

    if threshold == "variance-aware":
        variances = [rkhs_variance(arm_samples[i], bandwidth) for i in range(N)]

    rows: list[int] = []
    cols: list[int] = []
    for i in range(N):
        for j in range(i + 1, N):
            stat = float(np.sqrt(mmd2_plugin(arm_samples[i], arm_samples[j], bandwidth)))
            if threshold == "variance-aware":
                b_ij = variance_aware_threshold(n, delta_prime, N, variances[i], variances[j], const=const)
            else:
                b_ij = subgaussian_threshold(n, delta_prime, N, const=const)
            if stat <= b_ij:
                rows.append(i)
                cols.append(j)

    adjacency = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(N, N))
    num_clusters, partition = connected_components(adjacency, directed=False)
    return partition, int(num_clusters)


def _fit(
    env,
    delta: float,
    budget_cap: int,
    rng: np.random.Generator,
    *,
    threshold: Literal["variance-aware", "subgaussian"],
    const: float = 1.0,
) -> Result:
    """Shared doubling-trick loop behind `fit`/`fit_sg`.

    K is assumed known (Algorithm 2's KABC(delta, K)) and read from `env.K`;
    it is used only in the stopping condition `num_clusters == K` and is
    never an argument here, since `fit`'s signature is fixed by CLAUDE.md's
    shared algorithm interface.

    `const` (default 1.0) is threaded through to `_cluster_once`'s threshold
    calls -- see its docstring. Not exposed via the public `fit`/`fit_sg`;
    scripts/calibrate_constants.py calls `_fit` directly to override it.
    """
    N = env.n_arms
    K = env.K

    tau, bandwidth, cap_result = _pilot_or_graceful_stop(env, N, delta, budget_cap, rng)
    if cap_result is not None:
        return cap_result

    partition = np.zeros(N, dtype=int)  # best-effort fallback: all arms in one cluster
    history: list[dict] = []
    stopped = False

    k = 0
    while True:
        k += 1
        if k > MAX_ITERATIONS:
            history.append({"stop_reason": STOP_REASON_MAX_ITERATIONS})
            break  # backstop only -- see MAX_ITERATIONS's docstring; budget_cap or the
            # memory guard should ordinarily have already stopped the loop well before this

        delta_k = iteration_confidence(k, delta)
        n_k = iteration_budget(k, N, delta_k)
        cost_k = N * n_k
        if tau + cost_k > budget_cap:
            history.append({"stop_reason": STOP_REASON_BUDGET_CAP})
            break  # cannot afford this iteration; report the last completed clustering

        try:
            new_partition, num_clusters = _cluster_once(
                env, N, n_k, delta_k, bandwidth, rng, threshold=threshold, const=const
            )
        except _MemoryGuardExceeded as e:
            history.append({"stop_reason": STOP_REASON_MEMORY_GUARD, "k": k, "n_k": n_k, "reason": str(e)})
            break  # report the last completed clustering; this iteration's cost is NOT
            # added to tau, matching the budget_cap path's convention of not charging
            # for an iteration that was never actually attempted

        partition = new_partition
        tau += cost_k

        history.append({"k": k, "n_k": n_k, "delta_k": delta_k, "tau": tau, "num_clusters": num_clusters})

        if num_clusters == K:
            stopped = True
            break

    return Result(partition=partition, tau=tau, stopped=stopped, history=history)
