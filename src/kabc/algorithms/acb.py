"""ACB* (Algorithm 2, arXiv:2406.11485, Thuot/Carpentier/Giraud/Verzelen 2024,
"Active clustering with bandit feedback") -- adaptive, mean-based active
clustering. Conforms to CLAUDE.md's standard fit(env, delta, budget_cap, rng)
interface: unlike KABC, the paper's own adaptive variant (ACB*, "theta* and
Delta* unknown") needs no oracle inputs, so no interface deviation is needed
here (contrast kabc-oracle, which does).

Deliberately does NOT import from kernels.py or thresholds.py: ACB is
mean-based (Euclidean distance between empirical means only), not kernel/MMD-
based. It has its own private distance estimator and sample-size formulas,
taken directly from the paper's Eq. 9-16 and Algorithm 1/2 (see docstrings
below for verbatim quotes of each).

K is assumed known, exactly as for kabc -- the paper's own "Conclusion and
limitations" states this explicitly ("we assumed that the number K of groups
was known to the learner") -- so `env.K` is reused the same way.

sigma^2 (the paper's Assumption 2.1 sub-Gaussian scale, assumed GIVEN in the
paper) is instead estimated from a small pilot phase here, per
docs/experiments.md's Exp 3 description ("sigma^2 estimated empirically per
arm") generalized to every experiment, for fairness against kabc's own
non-oracle bandwidth estimation. This pilot is unrelated to CLAUDE.md's
"non-negotiable" n_pilot=5 rule (that rule is specifically about KABC's
median-heuristic bandwidth) -- it's a separate, analogous pilot reusing the
same n_pilot=5 convention. Its cost counts toward tau, same principle.

The leading numerical constants c1..c6 in Eq. 9-16 are never resolved to
explicit numbers anywhere in the paper (checked the full text, including the
appendix's Lemma C.2/C.3 and Remark C.4 restatements: they remain symbolic,
"choose c large enough" proof constants). docs/experiments.md's calibration
protocol (section 0.2) explicitly anticipates this: "Same protocol applied
independently to acb and tas-fw so no method is disadvantaged."

c1..c6 are NOT structurally interchangeable, so they are kept as six
separately-named constants here (each defaulting to 1.0), not collapsed to
one shared value: c1/c2 (Eq. 10, n_s) are a dimension-free vs. sqrt(d)-scaling
pair from Lemma C.2's proof of SRI; c3 (Eq. 11, n_max) is a third, distinct
sqrt(d)-scaling constant from the same lemma but a different sub-argument
(the final per-representative precision, not the doubling-test schedule);
c4/c5 (Eq. 13, J and I) are the analogous dimension-free vs. sqrt(d)-scaling
pair from Lemma C.3's proof of ADC -- structurally similar to c1/c2 but from
a different lemma, never claimed equal to them; c6 (Eq. 16, n'_p) is a sixth,
distinct constant multiplying a combined (dimension-free + sqrt(d)) term as a
single sum, from Algorithm 2's outer-loop refinement (outside Lemma C.2/C.3
entirely -- referenced only via "(55)" in the appendix). Collapsing all six
to one shared constant would artificially force the low-dimensional and
high-dimensional terms to scale identically with d, which matters concretely
here: docs/experiments.md's Exp 1 explicitly sweeps d in {2,10,50,100} and
asks "does the nonparametric price grow with d?" -- exactly the
low-dim/high-dim crossover these constants jointly control.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from kabc.algorithms import Result

N_PILOT = 5
_MAX_SCAN_LEVELS = 1000  # defensive circuit-breaker; budget exhaustion is the real stop condition


class _BudgetExhausted(Exception):
    """Internal signal: budget_cap can't afford the next sample batch. Unwinds
    ACB*'s deep loop nesting (scan level l -> p -> SRI's candidate loop u ->
    doubling-test loop s) cleanly, without threading a break-flag through
    four loop levels."""


class _Budget:
    def __init__(self, cap: int):
        self.cap = cap
        self.tau = 0

    def afford(self, cost: int) -> bool:
        return self.tau + cost <= self.cap

    def spend(self, cost: int) -> None:
        self.tau += cost


def fit(env, delta: float, budget_cap: int, rng: np.random.Generator) -> Result:
    """ACB*: adaptive active clustering (Algorithm 2, "ACB* (theta* and Delta*
    unknown)"). Mean-based baseline -- expected to be competitive with kabc on
    mean-separated instances (Exp 1) and mean-blind (chance-level partition)
    on same-mean, different-shape instances (Exp 2), by construction: it only
    ever looks at estimated means.
    """
    return _fit(env, delta, budget_cap, rng, c1=1.0, c2=1.0, c3=1.0, c4=1.0, c5=1.0, c6=1.0)


def _pull_batch(env, arm: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([np.atleast_1d(np.asarray(env.pull(arm, rng), dtype=float)) for _ in range(n)])


def _paired_empirical_means(
    env, arm: int, n: int, rng: np.random.Generator, budget: _Budget
) -> tuple[np.ndarray, np.ndarray]:
    """empirical_mean(a, n) (p.6): two INDEPENDENT empirical means, each from
    n fresh samples (2n pulls total) -- used everywhere in place of a single
    empirical mean to get the unbiased distance estimator below.
    """
    cost = 2 * n
    if not budget.afford(cost):
        raise _BudgetExhausted
    mu = _pull_batch(env, arm, n, rng).mean(axis=0)
    mu_prime = _pull_batch(env, arm, n, rng).mean(axis=0)
    budget.spend(cost)
    return mu, mu_prime


def _dhat_sq(
    mu_a: np.ndarray, mu_a_prime: np.ndarray, mu_b: np.ndarray, mu_b_prime: np.ndarray
) -> float:
    """Unbiased estimator of ||mu_a - mu_b||^2 (p.6, "Estimating distances"):
    d_hat^2_ab = <mu_hat_a - mu_hat_b, mu_hat'_a - mu_hat'_b>. Can be negative
    under noise when the true distance is 0 or small (same structure as
    kernels.mmd2_unbiased's known negativity near the null).
    """
    return float(np.dot(mu_a - mu_b, mu_a_prime - mu_b_prime))


def _estimate_sigma_sq(
    env, N: int, rng: np.random.Generator, budget: _Budget
) -> tuple[float, int]:
    """Pilot estimate of sigma^2 (Assumption 2.1's max_a||Sigma_a||_op bound)
    and the observation dimension d, both needed by every formula below.
    Per-arm, per-coordinate sample variance from N_PILOT pulls; the max over
    arms and coordinates is a practical proxy for max_a||Sigma_a||_op, exact
    when Sigma_a is isotropic (true of every environment in this repo).
    """
    cost = N * N_PILOT
    if not budget.afford(cost):
        raise _BudgetExhausted
    max_var = 0.0
    d = 1
    for i in range(N):
        X = _pull_batch(env, i, N_PILOT, rng)
        d = X.shape[1]
        max_var = max(max_var, float(np.var(X, axis=0, ddof=1).max()))
    budget.spend(cost)
    return max(max_var, 1e-12), d  # floor guards a degenerate all-identical pilot


def _represented_test(
    mu_a: np.ndarray,
    mu_a_prime: np.ndarray,
    representatives: list[tuple[np.ndarray, np.ndarray]],
    Delta: float,
) -> bool:
    """REPRESENTED_TEST (p.6): True iff a is (likely) already represented in S,
    i.e. min_{b in S} <mu_hat_a - mu_hat_b, mu_hat'_a - mu_hat'_b> <= Delta^2/2.
    """
    return min(_dhat_sq(mu_a, mu_a_prime, mu_b, mu_b_prime) for mu_b, mu_b_prime in representatives) <= (
        Delta**2 / 2.0
    )


def _sri_schedule(
    delta: float,
    Delta: float,
    theta: float,
    K: int,
    d: int,
    sigma_sq: float,
    c1: float,
    c2: float,
    c3: float,
):
    """U, r, n_s(.), s0, n_max, T_max per Eq. 9-12 (p.6-7)."""
    U = math.ceil(8.0 / theta * math.log(8.0 * K / delta))
    r = max(1, math.ceil(math.log2(math.log(4.0 * U / delta))))

    def n_s(s: int) -> int:
        # Eq. 10 uses 2**s (exponential -- "roughly doubling sample size", per
        # the prose right after Eq. 12), not 2*s. Verified against the
        # rendered PDF, not just the flat text extraction, since 2^s vs 2s is
        # exactly the kind of superscript that plain text extraction loses.
        term1 = c1 * sigma_sq / Delta**2 * (2.0**s + math.log(12.0 * K))
        term2 = c2 * sigma_sq / Delta**2 * math.sqrt(d * (2.0**s + math.log(6.0)))
        return math.ceil(max(term1, term2))

    s0 = r
    for s in range(1, r + 1):
        if n_s(s) >= 2:
            s0 = min(r, s)
            break

    n_max = max(n_s(r), math.ceil(c3 * sigma_sq / Delta**2 * math.sqrt(d) * math.log(2.0 * K)))

    tail = range(s0 + 1, r + 1)
    T_max = (
        2 * K * (n_max + sum(n_s(s) for s in tail))
        + 2 * U * n_s(s0)
        + 2 * U * sum(n_s(s) / 2.0 ** (s - 4) for s in tail)
    )
    return U, r, s0, n_s, n_max, math.ceil(T_max)


def _sri(
    env,
    N: int,
    K: int,
    d: int,
    delta: float,
    Delta: float,
    theta: float,
    sigma_sq: float,
    c1: float,
    c2: float,
    c3: float,
    rng: np.random.Generator,
    budget: _Budget,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """SRI procedure (p.7, Sequential Representative Identification): returns
    {arm: (mu_hat, mu_hat')} for up to K representatives, one per group.
    """
    U, r, s0, n_s, n_max, T_max = _sri_schedule(delta, Delta, theta, K, d, sigma_sq, c1, c2, c3)

    a0 = int(rng.integers(N))
    representatives: dict[int, tuple[np.ndarray, np.ndarray]] = {
        a0: _paired_empirical_means(env, a0, n_max, rng, budget)
    }
    sri_spent = 2 * n_max

    for _ in range(U):
        if len(representatives) == K or sri_spent > T_max:
            break
        au = int(rng.integers(N))
        rejected = False
        for s in range(s0, r + 1):
            mu_a, mu_a_prime = _paired_empirical_means(env, au, n_s(s), rng, budget)
            sri_spent += 2 * n_s(s)
            if _represented_test(mu_a, mu_a_prime, list(representatives.values()), Delta):
                rejected = True
                break
        if not rejected:
            representatives[au] = _paired_empirical_means(env, au, n_max, rng, budget)
            sri_spent += 2 * n_max

    return representatives


def _adc(
    env,
    N: int,
    K: int,
    d: int,
    delta: float,
    Delta: float,
    representatives: dict[int, tuple[np.ndarray, np.ndarray]],
    sigma_sq: float,
    c4: float,
    c5: float,
    rng: np.random.Generator,
    budget: _Budget,
) -> np.ndarray:
    """ADC procedure (p.7, Active Distance-based Classification): re-estimate
    each representative's mean precisely (J samples), then classify every
    other arm by nearest estimated representative mean (I samples), per
    Eq. 13-14. Returns an (N,) integer partition (labels 0..K-1).
    """
    L = math.log(6.0 * N * K / delta)
    J = math.ceil(
        max(c4 * sigma_sq / Delta**2 * L, c5 * sigma_sq / Delta**2 * math.sqrt(d * N / (K * L)))
    )
    I = math.ceil(
        max(c4 * sigma_sq / Delta**2 * L, c5 * sigma_sq / Delta**2 * math.sqrt(d * K / (N * L)))
    )

    rep_arms = list(representatives.keys())
    group_means = {b: _paired_empirical_means(env, b, J, rng, budget) for b in rep_arms}

    partition = np.full(N, -1, dtype=int)
    for j, b in enumerate(rep_arms):
        partition[b] = j

    for a in range(N):
        if a in group_means:
            continue
        mu_a, mu_a_prime = _paired_empirical_means(env, a, I, rng, budget)
        dists = [_dhat_sq(mu_a, mu_a_prime, *group_means[b]) for b in rep_arms]
        partition[a] = int(np.argmin(dists))

    return partition


def _fit(
    env,
    delta: float,
    budget_cap: int,
    rng: np.random.Generator,
    *,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 1.0,
    c4: float = 1.0,
    c5: float = 1.0,
    c6: float = 1.0,
) -> Result:
    """ACB* (Algorithm 2, p.8): scan increasing "levels" l = 0, 1, ...; within
    each, scan candidate separation/imbalance pairs (Delta_p, theta_{p,l}) for
    p = 0,...,l (Eq. 15-16); call SRI at each; the first time SRI finds
    exactly K representatives, refine their means, estimate the separation
    Delta_hat among them, and hand off to ADC for the final classification.
    """
    if not 0.0 < delta <= 1.0:
        raise ValueError(f"delta must be in (0, 1], got {delta!r}")
    if budget_cap < 1:
        raise ValueError(f"budget_cap must be >= 1, got {budget_cap!r}")

    N = env.n_arms
    K = env.K
    budget = _Budget(cap=budget_cap)
    partition = np.zeros(N, dtype=int)  # best-effort fallback: all arms in one cluster
    history: list[dict] = []

    pilot_cost = N * N_PILOT
    if budget_cap < pilot_cost:
        warnings.warn(
            f"budget_cap={budget_cap} cannot afford ACB's sigma^2 pilot phase "
            f"(N*n_pilot={pilot_cost}, N={N}, n_pilot={N_PILOT}); returning stopped=False "
            f"with no pulls made (tau=0)",
            stacklevel=2,
        )
        return Result(partition=partition, tau=0, stopped=False, history=[])

    try:
        sigma_sq, d = _estimate_sigma_sq(env, N, rng, budget)

        Delta0_sq = sigma_sq * (math.log(K) + math.sqrt(d) + math.log(math.log(6.0 * N / delta)))

        for l in range(_MAX_SCAN_LEVELS):
            for p in range(l + 1):
                delta_l = delta / (6.0 * (l + 1) ** 3)
                theta_pl = max(1.0 / (K * 2.0 ** (l - p)), 1.0 / N)
                Delta_p = math.sqrt(Delta0_sq / 2.0**p)
                n_p_prime = math.ceil(
                    c6
                    * sigma_sq
                    / Delta_p**2
                    * (math.log(3.0 * K**2 / delta) + math.sqrt(d * math.log(3.0 * K**2 / delta)))
                )

                representatives = _sri(
                    env, N, K, d, delta_l, Delta_p, theta_pl, sigma_sq, c1, c2, c3, rng, budget
                )
                history.append(
                    {
                        "l": l,
                        "p": p,
                        "Delta_p": Delta_p,
                        "theta_pl": theta_pl,
                        "num_representatives": len(representatives),
                        "tau": budget.tau,
                    }
                )

                if len(representatives) == K:
                    refined = {a: _paired_empirical_means(env, a, n_p_prime, rng, budget) for a in representatives}
                    Delta_hat_sq = min(
                        _dhat_sq(*refined[a], *refined[b])
                        for a in refined
                        for b in refined
                        if a != b
                    )
                    # Delta_hat^2 is an unbiased (possibly negative) estimator;
                    # floor at Delta_p^2 since SRI's success at this level
                    # already certifies true separation >= Delta_p.
                    Delta_hat = math.sqrt(max(Delta_hat_sq, Delta_p**2))

                    partition = _adc(
                        env, N, K, d, delta / 3.0, Delta_hat / math.sqrt(2.0), representatives,
                        sigma_sq, c4, c5, rng, budget,
                    )
                    return Result(partition=partition, tau=budget.tau, stopped=True, history=history)
    except _BudgetExhausted:
        pass

    return Result(partition=partition, tau=budget.tau, stopped=False, history=history)
