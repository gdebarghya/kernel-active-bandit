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
        return Result(partition=np.zeros(N, dtype=int), tau=tau, stopped=False, history=[])

    partition, num_clusters = _cluster_once(
        env, N, n_star, delta, bandwidth, rng, threshold="variance-aware"
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
        return 0, None, Result(partition=np.zeros(N, dtype=int), tau=0, stopped=False, history=[])

    # Pilot phase: n_pilot pulls/arm, pooled, for the median-heuristic bandwidth.
    # Counts toward tau; discarded afterward (not reused by the main loop).
    pilot = [_pull_batch(env, i, N_PILOT, rng) for i in range(N)]
    bandwidth = median_heuristic_bandwidth(np.concatenate(pilot, axis=0))
    return pilot_cost, bandwidth, None


def _cluster_once(
    env,
    N: int,
    n: int,
    delta_prime: float,
    bandwidth: float,
    rng: np.random.Generator,
    *,
    threshold: Literal["variance-aware", "subgaussian"],
) -> tuple[np.ndarray, int]:
    """One call of Algorithm 1's CLUSTER(n, delta'): draw n fresh samples/arm,
    build the same-cluster graph via the pairwise threshold, return
    (partition, num_clusters) as the graph's connected components.
    """
    arm_samples = [_pull_batch(env, i, n, rng) for i in range(N)]

    if threshold == "variance-aware":
        variances = [rkhs_variance(arm_samples[i], bandwidth) for i in range(N)]

    rows: list[int] = []
    cols: list[int] = []
    for i in range(N):
        for j in range(i + 1, N):
            stat = float(np.sqrt(mmd2_plugin(arm_samples[i], arm_samples[j], bandwidth)))
            if threshold == "variance-aware":
                b_ij = variance_aware_threshold(n, delta_prime, N, variances[i], variances[j])
            else:
                b_ij = subgaussian_threshold(n, delta_prime, N)
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
) -> Result:
    """Shared doubling-trick loop behind `fit`/`fit_sg`.

    K is assumed known (Algorithm 2's KABC(delta, K)) and read from `env.K`;
    it is used only in the stopping condition `num_clusters == K` and is
    never an argument here, since `fit`'s signature is fixed by CLAUDE.md's
    shared algorithm interface.
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
        delta_k = iteration_confidence(k, delta)
        n_k = iteration_budget(k, N, delta_k)
        cost_k = N * n_k
        if tau + cost_k > budget_cap:
            break  # cannot afford this iteration; report the last completed clustering

        partition, num_clusters = _cluster_once(env, N, n_k, delta_k, bandwidth, rng, threshold=threshold)
        tau += cost_k

        history.append({"k": k, "n_k": n_k, "delta_k": delta_k, "tau": tau, "num_clusters": num_clusters})

        if num_clusters == K:
            stopped = True
            break

    return Result(partition=partition, tau=tau, stopped=stopped, history=history)
