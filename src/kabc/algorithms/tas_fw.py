"""TaS-FW (Algorithm 1, arXiv:2409.05072, Yavas/Huang/Tan/Scarlett, "A General
Framework for Clustering and Distribution Matching with Bandit Feedback") --
Track-and-Stop based on Frank-Wolfe, for N-ary clustering of arms on a common
FINITE alphabet. Only run on Experiment 3's reduced instance (see
docs/experiments.md): its finite-alphabet assumption holds natively there
(MovieLens ratings on {1..5}); it is not run on Experiment 1's continuous
Gaussian instances (would need an off-label discretization approximation
outside the algorithm's proven guarantees -- deliberately not implemented
here, per that decision).

NAMING REMAP -- read this before cross-referencing the paper: the paper uses
K for the number of ARMS and M for the number of CLUSTERS. This repo's
existing convention (kabc.py, acb.py, environments.py's `Environment.K`) is
the OPPOSITE: N = number of arms (`env.n_arms`), K = number of clusters
(`env.K`). This module follows the REPO's convention (N=arms, K=clusters) in
all code and comments, NOT the paper's -- so every direct formula reference
below silently maps the paper's K -> this module's N, and the paper's M ->
this module's K. Getting this backwards is an easy way to introduce a subtle
bug, so it's called out here once, prominently, rather than per-formula.

Only the M=K, no-unconstrained-group case is implemented: every hypothesis
sigma is a partition of all N arms into EXACTLY K clusters, each of size >=2
-- matching every environment in this repo (all arms belong to some true
cluster, no leftover/unconstrained arms), and matching the paper's own stated
simplification ("Fixing the number of clusters to the same value for all
hypotheses... simplifies the presentation", Section II).

WHY THIS IS EXPENSIVE (per docs/experiments.md: "small instances only, N<=20"
and Exp 3's "orders of magnitude more compute per decision"). Two compounding
costs, both paid at EVERY time step until stopping:
1. The Frank-Wolfe direction (Eq. 50 / Algorithm 1 line 6) reduces to a
   single linear program (Eq. 54) over |B| <= |C| columns -- this part is,
   as the paper states, "computationally efficient... only requires solving
   a single linear program at each time step."
2. But BOTH sigma_hat(t) (Eq. 45, an argmin over ALL of C) and the LP's own
   constraint set B (Eq. 48/52, built from evaluating g^sigma' over candidate
   hypotheses) require scanning the FULL hypothesis space C. |C| here is the
   number of ways to partition N arms into exactly K non-empty clusters --
   a Stirling number of the second kind, S(N, K) -- which grows
   combinatorially in N for fixed K, not polynomially: S(6,3) = 90 (Exp 3's
   reduced instance -- trivial) but S(15,3) ~ 2.37 million (the N=15 scale
   once considered for Exp 1 -- intractable per-time-step, which is exactly
   why that subset was dropped rather than worked around).

Deliberately does NOT import from kernels.py or thresholds.py: like acb.py,
this is a finite-alphabet, information-theoretic (KL-divergence-based)
method, unrelated to KABC's kernel/MMD machinery.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from kabc.algorithms import Result


def fit(env, delta: float, budget_cap: int, rng: np.random.Generator) -> Result:
    """TaS-FW (Algorithm 1). `env.pull` must return values from a small,
    common finite alphabet (e.g. MovieLens ratings on {1,...,5}) -- there is
    no discretization step here; see the module docstring for why.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")
    if budget_cap < 1:
        raise ValueError(f"budget_cap must be >= 1, got {budget_cap!r}")

    N = env.n_arms
    K = env.K
    hypotheses = _enumerate_hypotheses(N, K)
    if not hypotheses:
        raise ValueError(f"no valid hypothesis (partition of {N} arms into {K} clusters of size >= 2)")

    # Initialization (Algorithm 1): sample each arm once.
    outcomes: list[list] = [[] for _ in range(N)]
    for i in range(N):
        outcomes[i].append(env.pull(i, rng))
    t = N
    if t > budget_cap:
        return _graceful_stop(N, t, [])

    alphabet = _infer_alphabet(outcomes)
    x_tilde = np.full(N, 1.0 / N)  # x~(K) = (1/K)*1 in the paper's N (arms)
    z_running_sum = x_tilde.copy() * N  # sum_{s=1}^{t} z~(s), so x~(t) = this / t

    history: list[dict] = []
    sigma_hat, Z_t = _current_best_and_stat(outcomes, hypotheses, alphabet, N)
    beta = _threshold(t, delta, K, len(alphabet), N)

    while Z_t < beta:
        if t >= budget_cap:
            return _graceful_stop(N, t, history, sigma_hat)

        t += 1
        r_t = t ** (-4.0 / 5.0)

        if _is_forced_exploration_step(t):
            z_t = np.full(N, 1.0 / N)
        else:
            z_t = _frank_wolfe_direction(outcomes, hypotheses, alphabet, N, sigma_hat, x_tilde, r_t)

        z_running_sum += z_t
        x_tilde = z_running_sum / t

        # C-tracking sampling rule (Eq. 57): pull the most "underserved" arm
        # relative to the running target allocation t*x~(t).
        counts = np.array([len(outcomes[i]) for i in range(N)], dtype=float)
        arm = int(np.argmax(t * x_tilde - counts))
        outcomes[arm].append(env.pull(arm, rng))

        alphabet = _infer_alphabet(outcomes)
        sigma_hat, Z_t = _current_best_and_stat(outcomes, hypotheses, alphabet, N)
        beta = _threshold(t, delta, K, len(alphabet), N)
        history.append({"t": t, "Z_t": Z_t, "beta": beta, "sigma_hat_clusters": len(sigma_hat)})

    partition = _hypothesis_to_partition(sigma_hat, N)
    return Result(partition=partition, tau=t, stopped=True, history=history)


def _graceful_stop(N: int, tau: int, history: list[dict], sigma_hat=None) -> Result:
    if sigma_hat is None:
        partition = np.zeros(N, dtype=int)  # best-effort fallback: all arms in one cluster
    else:
        partition = _hypothesis_to_partition(sigma_hat, N)
    return Result(partition=partition, tau=tau, stopped=False, history=history)


# --------------------------------------------------------------------------
# Hypothesis space C: partitions of [N] arms into exactly K clusters, |.|>=2
# --------------------------------------------------------------------------

def _enumerate_hypotheses(N: int, K: int) -> list[tuple[frozenset, ...]]:
    """All set partitions of range(N) into exactly K non-empty blocks, each
    of size >= 2 ("a cluster contains at least two arms"; no unconstrained
    group here since every environment's true partition covers all arms).
    |C| = S(N,K), a Stirling number of the second kind restricted to blocks
    of size >= 2 -- see the module docstring for why this is the expensive part.
    """
    hypotheses = []
    for partition in _set_partitions(list(range(N))):
        if len(partition) == K and all(len(block) >= 2 for block in partition):
            hypotheses.append(tuple(frozenset(block) for block in partition))
    return hypotheses


def _set_partitions(items: Sequence[int]):
    """Yield all set partitions of `items` as lists of lists (standard
    recursive restricted-growth-string generation)."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for smaller in _set_partitions(rest):
        # add `first` to each existing block, one at a time
        for i in range(len(smaller)):
            yield smaller[:i] + [[first] + smaller[i]] + smaller[i + 1 :]
        # or start a new block with just `first`
        yield [[first]] + smaller


# --------------------------------------------------------------------------
# G function (Eq. 16), g^sigma_P (Eq. 27), sigma_hat/Z(t) (Eq. 45, 62-63)
# --------------------------------------------------------------------------

def _infer_alphabet(outcomes: list[list]) -> list:
    values = set()
    for arm_outcomes in outcomes:
        values.update(arm_outcomes)
    return sorted(values)


def _empirical_dist(arm_outcomes: list, alphabet: list) -> np.ndarray:
    counts = Counter(arm_outcomes)
    n = len(arm_outcomes)
    return np.array([counts.get(a, 0) / n for a in alphabet], dtype=float)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """D(P||Q) (Eq. 6), 0 * log(0/.) treated as 0 by convention."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _g_function(dists: list[np.ndarray], weights: list[float]) -> tuple[float, np.ndarray]:
    """G(P_A, w_A) (Eq. 16-17). Returns (value, pooled distribution W) since
    W is needed again by the gradient computation."""
    total_w = sum(weights)
    if total_w == 0:
        return 0.0, dists[0] if dists else np.array([])
    w_arr = np.array(weights)
    pooled = sum(w * d for w, d in zip(weights, dists)) / total_w
    value = sum(w * _kl(d, pooled) for w, d in zip(weights, dists))
    return float(value), pooled


def _g_sigma(
    empirical_dists: list[np.ndarray], w: np.ndarray, hypothesis: tuple[frozenset, ...]
) -> float:
    """g^sigma_P(w) (Eq. 27): sum over clusters of G(.)."""
    total = 0.0
    for cluster in hypothesis:
        idx = sorted(cluster)
        total += _g_function([empirical_dists[i] for i in idx], [w[i] for i in idx])[0]
    return total


def _current_best_and_stat(
    outcomes: list[list], hypotheses: list[tuple[frozenset, ...]], alphabet: list, N: int
) -> tuple[tuple[frozenset, ...], float]:
    """sigma_hat(t) (Eq. 45) and Z(t) = t * (second-smallest g^sigma) (Eq. 62-63)."""
    t = sum(len(o) for o in outcomes)
    w = np.array([len(o) for o in outcomes], dtype=float) / t
    empirical_dists = [_empirical_dist(o, alphabet) for o in outcomes]

    scores = [(_g_sigma(empirical_dists, w, h), h) for h in hypotheses]
    scores.sort(key=lambda item: item[0])
    sigma_hat = scores[0][1]
    second_smallest = scores[1][0] if len(scores) > 1 else scores[0][0]
    return sigma_hat, t * second_smallest


def _threshold(t: int, delta: float, K: int, alphabet_size: int, N: int) -> float:
    """beta(t,delta) (Eq. 66). K_tilde (Eq. 64) is the max number of arms
    that belong to a cluster across all hypotheses -- with M=K fixed and no
    unconstrained group here, every hypothesis clusters all N arms, so
    K_tilde = N always.
    """
    K_tilde = N
    return (
        math.log(1.0 / delta)
        + (K * alphabet_size + K_tilde + 2) * math.log(t + 1)
        + math.log(math.pi**2 / 6.0 - 1.0)
    )


# --------------------------------------------------------------------------
# Frank-Wolfe direction z(t): the payoff matrix (Eq. 53) and LP (Eq. 54)
# --------------------------------------------------------------------------

def _g_sigma_gradient(
    empirical_dists: list[np.ndarray], w: np.ndarray, hypothesis: tuple[frozenset, ...], N: int
) -> np.ndarray:
    """Gradient of g^sigma_P(w) w.r.t. w, needed for the payoff matrix (Eq. 53).
    Within a cluster A with pooled distribution W_A, d/dw_i [sum_{j in A} w_j
    D(P_j||W_A)] = D(P_i||W_A) (the cross-terms from W_A's own w-dependence
    cancel by Lemma 2's first-order optimality of W_A as the weighted mean).
    Arms outside any cluster of this hypothesis don't appear in g^sigma, so
    their gradient entries are 0 (there is no unconstrained group here, but
    keep this well-defined regardless).
    """
    grad = np.zeros(N)
    for cluster in hypothesis:
        idx = sorted(cluster)
        _, pooled = _g_function([empirical_dists[i] for i in idx], [w[i] for i in idx])
        for i in idx:
            grad[i] = _kl(empirical_dists[i], pooled)
    return grad


def _frank_wolfe_direction(
    outcomes: list[list],
    hypotheses: list[tuple[frozenset, ...]],
    alphabet: list,
    N: int,
    sigma_hat: tuple[frozenset, ...],
    x_prev: np.ndarray,
    r_t: float,
) -> np.ndarray:
    """z(t) (Eq. 50, Algorithm 1 line 6): solve the LP (Eq. 54) over the
    r-subdifferential's generating hypotheses B (Eq. 48/52).
    """
    total = sum(len(o) for o in outcomes)
    w = np.array([len(o) for o in outcomes], dtype=float) / total
    empirical_dists = [_empirical_dist(o, alphabet) for o in outcomes]

    others = [h for h in hypotheses if h != sigma_hat]
    g_values = {h: _g_sigma(empirical_dists, w, h) for h in others}
    G_sigma = min(g_values.values())

    # B = hypotheses within r_t of the current min (Eq. 48/52).
    B = [h for h in others if g_values[h] < G_sigma + r_t]
    if not B:
        B = [min(others, key=lambda h: g_values[h])]

    gradients = [_g_sigma_gradient(empirical_dists, w, h, N) for h in B]
    # Payoff matrix M[k, b] = <e_k - x_prev, grad_b> (Eq. 53).
    M = np.array([[grad[k] - np.dot(x_prev, grad) for grad in gradients] for k in range(N)])

    # LP (Eq. 54): max_{z in Simplex_N, u} u  s.t. (z^T M)_b >= u for b in B.
    # linprog minimizes, over variables [z_1..z_N, u]; z^T M has shape (|B|,).
    n_vars = N + 1
    c = np.zeros(n_vars)
    c[-1] = -1.0  # maximize u == minimize -u
    A_ub = np.zeros((len(B), n_vars))
    A_ub[:, :N] = -M.T  # -(z^T M)_b + u <= 0  <=>  (z^T M)_b >= u
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(len(B))
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * N + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        # Numerically degenerate LP (can happen very early with few samples);
        # fall back to uniform exploration for this step rather than crash.
        return np.full(N, 1.0 / N)
    return result.x[:N]


def _is_forced_exploration_step(t: int) -> bool:
    """t in I_f (Eq. 55): ceil(sqrt(t)*log(t)) == ceil(sqrt(t+1)*log(t+1)) - 1."""
    if t < 2:
        return True
    return math.ceil(math.sqrt(t) * math.log(t)) == math.ceil(math.sqrt(t + 1) * math.log(t + 1)) - 1


# --------------------------------------------------------------------------
# Convert a hypothesis (tuple of frozensets) into a Result-style partition
# --------------------------------------------------------------------------

def _hypothesis_to_partition(hypothesis: tuple[frozenset, ...], N: int) -> np.ndarray:
    partition = np.zeros(N, dtype=int)
    for label, cluster in enumerate(hypothesis):
        for arm in cluster:
            partition[arm] = label
    return partition
