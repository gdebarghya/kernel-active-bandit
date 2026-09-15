"""ACB-Gaussian (Appendix A "Variant of the procedure and parametrization of
ACB", arXiv:2406.11485, Thuot/Carpentier/Giraud/Verzelen 2024) -- the
Gaussian-specialized, NON-adaptive ACB variant the paper's own authors use
for their Figure 1 numerical experiment (the one that gets <=1% error at
delta=0.1), not the main-text ACB*/Algorithm 2 that acb.py implements. See
acb.py's own module docstring for why acb.py exists and is untouched by this
file -- the two are deliberately separate baselines, not variants of a
shared function.

WHY A SEPARATE FILE, NOT A FLAG ON acb.py: this variant is fundamentally
non-adaptive (paper's own words, Appendix A: "we assume that Delta* is
known, and we implement the non-adaptive version of ACB") -- Algorithm 1
(SRI+ADC called once, with known Delta/theta), not Algorithm 2's adaptive
multiscale search over unknown (Delta*, theta*). It also replaces the
main-text's concentration-inequality-derived thresholds/sample sizes
(Eq. 9-13, constants c1-c6) with EXACT quantile-based ones (a chi-squared
quantile for n_max, a normal quantile plus a Monte-Carlo product-quantile
for I/J) -- calibration-free by construction, see the CALIBRATION note
below. Folding this into acb.py's `_fit` would require threading a third
code path through every one of SRI/ADC's formula sites; a separate file
mirroring acb.py's existing SRI/ADC decomposition is far less error-prone
to verify formula-by-formula against the paper (every formula below is
quoted from a direct read of the rendered PDF, page 13, same verification
standard as tas_fw.py's own NAMING REMAP note).

INTERFACE: `fit(env, delta, budget_cap, rng)` matches CLAUDE.md's standard
signature in call SHAPE only -- Delta, theta, and sigma are NOT passed as
parameters; they are read directly off `env` (`.Delta`, `.K`, `.sigma`),
attributes that only `kabc.environments.GaussianMixtureEnv` exposes. This is
NOT an oracle-cheat for the Experiment 1 comparison specifically:
GaussianMixtureEnv is CONSTRUCTED from these exact values (configs/exp1.yaml
passes Delta/sigma straight to the environment's own constructor), so
nothing is being estimated from data that a fair comparison would require
estimating -- Delta and theta genuinely are known at instance-construction
time, in the same sense N, K, and d already are to every algorithm in this
repo (see acb.py's own "K is assumed known" note). But this DOES mean
acb-gaussian is NOT a generic, drop-in registry entry the way acb.fit/
kabc.fit are: it is evaluated with Delta, theta known, matching the
non-adaptive Algorithm 1 setting the paper's own authors use for their
Figure 1 experiment -- not the fully adaptive ACB* -- since this is the
fairest, most literal reproduction of the comparison the paper's own
authors ran. State this explicitly wherever acb-gaussian's Experiment 1
numbers are reported (docs/results_exp1_exp2_exp3.tex's Baselines section,
docs/experiments.md's algorithm table) so a reader does not have to wonder
why this one baseline gets privileged information the others don't without
being told why. It hard-fails immediately (`_require_gaussian_env` below)
on any other environment (ScalePairEnv, MomentMatchedMixtureEnv,
RingBlobEnv, MovieLensPairEnv) rather than silently misapplying a
Gaussian-only method outside its valid domain -- this repo's Experiment
1 only.

CALIBRATION: none, deliberately -- see configs/calibration.yaml's own note
and docs/experiments.md's algorithm table entry for this algorithm. Every
constant here (n_max via a chi-squared quantile; I/J via a normal quantile
beta and a Monte-Carlo product-quantile alpha) is an EXACT delta-level
quantile of the statistic's true finite-sample null distribution, not a
concentration-inequality bound with a leading constant that has slack to
tighten empirically. There is nothing left to calibrate here; c_cal-style
empirical tightening would break the delta-PAC guarantee rather than trade
conservativeness for a tighter budget, unlike kabc/kabc-sg/acb's thresholds.

NO -0.5x SELF-COMPARISON ROW: every OTHER base algorithm in Experiment 1
(kabc, kabc-sg, acb) gets a `-0.5x` row (budget_cap = min(0.5*tau_own,
budget_cap)) specifically to measure how much MARGIN that algorithm's own
stopping rule carries beyond what a correct answer actually needed --
margin that exists BECAUSE their thresholds are conservative concentration-
inequality bounds with real slack (see the CALIBRATION note above and
c_cal's own role). acb-gaussian has no such slack to measure: its stopping
point IS the exact delta-level quantile, by construction, so a `-0.5x` row
here would not be testing "how conservative is this threshold" the way it
does for the other three -- it would just show that halving a budget that
was never conservative in the first place breaks the algorithm, a
different and much less informative question. This is not an oversight;
do not add one without first re-deriving what such a row would actually be
measuring for THIS algorithm.

INTERPRETIVE NOTE on the outer candidate-scan bound: the paper's Appendix A
text redefines n_s, n_max, I, J and the classifier for the Gaussian setting,
and explicitly drops the T_max/Line-12 early-stop check ("we do not limit
... the budget of SRI ... The condition from Line 12 in SRI is indeed not
used"), but does not redefine U, SRI's outer per-candidate-scan bound
(Eq. 9: U = ceil(8/theta * log(8K/delta))). This module reuses that same
U as the epoch limit, since nothing in Appendix A suggests otherwise and it
is the only U defined anywhere in the paper.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from kabc.algorithms import Result
from kabc.environments import GaussianMixtureEnv

_MC_SAMPLES_ALPHA = 200_000  # Monte Carlo sample count for the product-quantile alpha (see _alpha_quantile)


class _BudgetExhausted(Exception):
    """Internal signal: budget_cap can't afford the next sample batch. Same
    role as acb.py's own exception of the same name (unwinds SRI/ADC's loop
    nesting cleanly); not shared with acb.py to keep the two modules
    independently readable, matching that file's own module boundary.
    """


class _Budget:
    def __init__(self, cap: int):
        self.cap = cap
        self.tau = 0

    def afford(self, cost: int) -> bool:
        return self.tau + cost <= self.cap

    def spend(self, cost: int) -> None:
        self.tau += cost


def _require_gaussian_env(env) -> None:
    """Hard safeguard: acb-gaussian's formulas (n_max via a Gaussian/chi-
    squared quantile; the REPRESENTEDTEST debiasing shift d*sigma^2*(1/n_s +
    1/n_max), itself derived from Gaussian-noise variance being EXACTLY
    known) are only valid for GaussianMixtureEnv, per Appendix A's own
    "generated with standard Gaussian noise" experimental setup. Raise
    immediately rather than silently returning a meaningless (or
    misleadingly good/bad) result on Experiment 2/3's non-Gaussian
    families.
    """
    if not isinstance(env, GaussianMixtureEnv):
        raise TypeError(
            f"acb_gaussian.fit is only valid for kabc.environments.GaussianMixtureEnv "
            f"(Appendix A's Gaussian-specialized variant relies on Delta, theta, and "
            f"sigma being exactly known quantities of a Gaussian instance) -- got "
            f"{type(env).__name__!r}. This algorithm is Experiment 1 only; do not wire "
            f"it into Experiment 2/3's non-Gaussian families."
        )


def _pull_batch(env, arm: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([np.atleast_1d(np.asarray(env.pull(arm, rng), dtype=float)) for _ in range(n)])


def _single_empirical_mean(env, arm: int, n: int, rng: np.random.Generator, budget: _Budget) -> np.ndarray:
    """empirical_mean(a, n): ONE empirical mean from n fresh samples -- no
    dual sampling, per Appendix A ("we avoid dual sampling ... to save a
    factor two in the budget").
    """
    if not budget.afford(n):
        raise _BudgetExhausted
    mu = _pull_batch(env, arm, n, rng).mean(axis=0)
    budget.spend(n)
    return mu


def _alpha_quantile(q: float, d: int, rng: np.random.Generator, n_mc: int = _MC_SAMPLES_ALPHA) -> float:
    """alpha: the q-quantile of <X, Y> for independent standard X, Y ~
    N(0, I_d) (Appendix A: "a product of independent standard N(0, I_d),
    that we had to compute empirically with Monte Carlo" -- no closed form,
    hence scipy.stats can't supply this one directly, unlike beta below).
    Uses `rng` (the same threaded generator every other stochastic step in
    this module uses), NOT a fresh internal seed, so this Monte Carlo step
    stays reproducible under this repo's existing seeding convention. Does
    NOT consume budget: this is a computational quantile lookup, not an
    environment pull, exactly analogous to scipy.stats.norm.ppf/chi2.ppf
    calls elsewhere in this module.
    """
    X = rng.standard_normal((n_mc, d))
    Y = rng.standard_normal((n_mc, d))
    products = np.einsum("ij,ij->i", X, Y)
    return float(np.quantile(products, q))


def _sri_gaussian(
    env,
    N: int,
    K: int,
    d: int,
    delta: float,
    Delta: float,
    theta: float,
    sigma: float,
    rng: np.random.Generator,
    budget: _Budget,
) -> dict[int, tuple[np.ndarray, int]]:
    """SRI, Gaussian-specialized (Appendix A). Returns {arm: (mu_hat, n_used)}
    for up to K representatives -- `n_used` is always n_max (Line 11's
    convention: a representative's own mean is estimated once, at n_max,
    when it is added to S) but is threaded through explicitly rather than
    assumed, so the debiasing shift below is unambiguous even if that
    convention is ever revisited.
    """
    sigma_sq = sigma**2
    x = stats.chi2.ppf(1.0 - delta / K, df=d)
    if x <= d:
        raise ValueError(
            f"chi2 quantile x={x!r} <= d={d!r} (delta={delta!r}, K={K!r}): n_max = "
            f"4*sigma^2/Delta^2*(x-d) would be non-positive -- delta/K too large for "
            f"this d to produce a meaningful (1-delta/K) quantile above the mean"
        )
    n_max = math.ceil(4.0 * sigma_sq / Delta**2 * (x - d))

    U_prime = (1.0 / theta) * math.log(1.0 / delta)
    n0 = max(1, math.ceil((K / U_prime) * n_max))
    r = max(0, math.ceil(math.log2(n_max / n0))) if n_max > n0 else 0

    def n_s(s: int) -> int:
        return math.ceil(2.0**s * n0)

    U = math.ceil(8.0 / theta * math.log(8.0 * K / delta))

    a0 = int(rng.integers(N))
    representatives: dict[int, tuple[np.ndarray, int]] = {
        a0: (_single_empirical_mean(env, a0, n_max, rng, budget), n_max)
    }

    for _ in range(U):
        if len(representatives) == K:
            break
        au = int(rng.integers(N))
        rejected = False
        for s in range(0, r + 1):
            ns = n_s(s)
            mu_au = _single_empirical_mean(env, au, ns, rng, budget)
            # REPRESENTEDTEST, Gaussian (Appendix A): plain (biased) squared
            # distance vs. representative's stored mean, compared against
            # Delta^2/2 PLUS the debiasing shift d*sigma^2*(1/ns + 1/n_max)
            # -- this shift is what corrects for the estimator's known bias
            # (E[||mu_hat_a - mu_bar_b||^2] = ||mu_a - mu_b||^2 + d*sigma^2*
            # (1/ns + 1/n_max) when a, b are in the same group), replacing
            # the main-text's unbiased-by-construction dual-sample estimator.
            threshold = Delta**2 / 2.0 + d * sigma_sq * (1.0 / ns + 1.0 / n_max)
            if min(float(np.sum((mu_au - mu_b) ** 2)) for mu_b, _ in representatives.values()) <= threshold:
                rejected = True
                break
        if not rejected:
            representatives[au] = (_single_empirical_mean(env, au, n_max, rng, budget), n_max)

    return representatives


def _adc_gaussian(
    env,
    N: int,
    K: int,
    d: int,
    delta: float,
    Delta: float,
    representatives: dict[int, tuple[np.ndarray, int]],
    sigma: float,
    rng: np.random.Generator,
    budget: _Budget,
) -> np.ndarray:
    """ADC, Gaussian-specialized (Appendix A): re-estimate each
    representative's mean precisely (J samples, single sampling), then
    classify every other arm by nearest estimated representative mean (I
    samples, single sampling, PLAIN squared distance -- no debiasing shift:
    unlike REPRESENTEDTEST's binary accept/reject against a fixed threshold,
    the classifier only needs the arg-min ordering across j, and the bias
    term added by single-sampling is identical (d*sigma^2*(1/I + 1/J)) for
    every candidate group j, so it cancels in the argmin and the paper
    correctly omits it here).
    """
    sigma_sq = sigma**2
    q = 1.0 - delta / (4.0 * K * (N - K))
    beta = stats.norm.ppf(q)
    alpha = _alpha_quantile(q, d, rng)

    I = math.ceil(sigma_sq / Delta**2 * max(16.0 * beta, 4.0 * math.sqrt(2.0 * K / N) * alpha))
    J = math.ceil(sigma_sq / Delta**2 * max(16.0 * beta, 4.0 * math.sqrt(2.0 * N / K) * alpha))

    rep_arms = list(representatives.keys())
    group_means = {b: _single_empirical_mean(env, b, J, rng, budget) for b in rep_arms}

    partition = np.full(N, -1, dtype=int)
    for j, b in enumerate(rep_arms):
        partition[b] = j

    for a in range(N):
        if a in group_means:
            continue
        mu_a = _single_empirical_mean(env, a, I, rng, budget)
        dists = [float(np.sum((mu_a - group_means[b]) ** 2)) for b in rep_arms]
        partition[a] = int(np.argmin(dists))

    return partition


def fit(env, delta: float, budget_cap: int, rng: np.random.Generator) -> Result:
    """ACB-Gaussian: non-adaptive ACB (Algorithm 1), Gaussian-specialized
    per Appendix A, with Delta/theta/sigma read from `env` (see this
    module's docstring for why that is legitimate here and not elsewhere).
    Experiment 1 (GaussianMixtureEnv) only -- see `_require_gaussian_env`.
    """
    _require_gaussian_env(env)
    if not 0.0 < delta <= 1.0:
        raise ValueError(f"delta must be in (0, 1], got {delta!r}")
    if budget_cap < 1:
        raise ValueError(f"budget_cap must be >= 1, got {budget_cap!r}")

    N = env.n_arms
    K = env.K
    d = env.d
    Delta = env.Delta
    sigma = env.sigma
    theta = 1.0 / K  # GaussianMixtureEnv._balanced_partition requires N % K == 0: EXACTLY balanced

    budget = _Budget(cap=budget_cap)
    partition = np.zeros(N, dtype=int)  # best-effort fallback: all arms in one cluster
    history: list[dict] = []

    try:
        representatives = _sri_gaussian(env, N, K, d, delta / 2.0, Delta, theta, sigma, rng, budget)
        history.append({"stage": "sri", "num_representatives": len(representatives), "tau": budget.tau})
        if len(representatives) == K:
            partition = _adc_gaussian(env, N, K, d, delta / 2.0, Delta, representatives, sigma, rng, budget)
            history.append({"stage": "adc", "tau": budget.tau})
            return Result(partition=partition, tau=budget.tau, stopped=True, history=history)
    except _BudgetExhausted:
        pass

    return Result(partition=partition, tau=budget.tau, stopped=False, history=history)
