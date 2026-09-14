import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from kabc.algorithms.baselines import STOP_REASON_DEGENERATE_BANDWIDTH, fit
from kabc.environments import GaussianMixtureEnv, ScalePairEnv


class _MostlyTiedEnv:
    """Test double: N-1 arms always pull an identical value, one arm pulls a
    different (but still identical-across-its-own-pulls) value -- same
    construction as tests/test_kabc.py's env of the same name, engineered so
    the pooled sample's MEDIAN pairwise distance is exactly zero (a majority
    of pairs are tied-value pairs) while the MEAN is not.
    """

    def __init__(self, N: int, K: int = 3):
        self.n_arms = N
        self.K = K
        self.true_partition = np.zeros(N, dtype=int)

    def pull(self, arm_index: int, rng: np.random.Generator) -> float:
        return 3.0 if arm_index == 0 else 5.0


class _FullyDegenerateEnv:
    """Test double: EVERY arm always pulls the identical value -- no
    bandwidth is recoverable from this data at all, even with the mean
    fallback."""

    def __init__(self, N: int, K: int = 3):
        self.n_arms = N
        self.K = K
        self.true_partition = np.zeros(N, dtype=int)

    def pull(self, arm_index: int, rng: np.random.Generator) -> float:
        return 5.0


def test_recovers_partition_on_well_separated_instance():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    rng = np.random.default_rng(0)
    result = fit(env, delta=0.05, budget_cap=1200, rng=rng)

    assert result.stopped is True
    assert result.tau == 1200
    assert adjusted_rand_score(env.true_partition, result.partition) == pytest.approx(1.0)


def test_same_mean_different_variance_depends_on_budget():
    # Unlike ACB (structurally mean-blind at ANY budget, kernels.py's own
    # kernel k-means uses the RBF kernel, which encodes shape/variance
    # information once n is large enough to represent it) -- so unif-kkm's
    # failure on same-mean instances is budget-DEPENDENT, not structural.
    # At small n it behaves near chance; at large n it can pick up the
    # variance difference. This is a behavioral note, not a strict claim
    # about exact ARI thresholds (kernel k-means has no formal guarantee
    # here), so only the large-n recovery is a hard assertion.
    env_small = ScalePairEnv(N=40, sigma_B=3.0)
    result_small = fit(env_small, delta=0.05, budget_cap=80, rng=np.random.default_rng(1))  # n=2
    ari_small = adjusted_rand_score(env_small.true_partition, result_small.partition)
    assert ari_small < 0.5  # near chance at low n; not a tight bound, just "hasn't recovered"

    env_large = ScalePairEnv(N=40, sigma_B=3.0)
    result_large = fit(env_large, delta=0.05, budget_cap=4000, rng=np.random.default_rng(1))  # n=100
    ari_large = adjusted_rand_score(env_large.true_partition, result_large.partition)
    assert ari_large == pytest.approx(1.0)


def test_stops_gracefully_when_cap_affords_no_pulls():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    rng = np.random.default_rng(0)

    with pytest.warns(UserWarning):
        result = fit(env, delta=0.05, budget_cap=5, rng=rng)  # budget_cap < N

    assert result.stopped is False
    assert result.tau == 0
    assert result.partition.shape == (12,)


def test_raises_on_invalid_budget_cap():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    with pytest.raises(ValueError):
        fit(env, delta=0.05, budget_cap=0, rng=np.random.default_rng(0))


def test_degenerate_bandwidth_falls_back_to_mean_and_still_completes():
    # Real incident (Experiment 3): a low-budget unif-kkm row (one of the
    # reduced-budget tau_kabc-derived variants) drew a pooled sample whose
    # pairwise distances were all (numerically) zero on a skewed-rating
    # MovieLens movie, crashing instead of degrading gracefully like
    # kabc.py's analogous pilot-phase fallback already does. Must fall back
    # to the mean pairwise distance and proceed normally.
    env = _MostlyTiedEnv(N=12, K=3)
    rng = np.random.default_rng(0)
    with pytest.warns(UserWarning, match="fell back to the mean pairwise distance"):
        result = fit(env, delta=0.05, budget_cap=1200, rng=rng)
    assert result.stopped is True
    assert result.history == []  # not the graceful degenerate-bandwidth stop -- recovered and clustered


def test_stop_reason_is_degenerate_bandwidth_when_mean_fallback_also_fails():
    # Every pooled sample is identical -- no bandwidth is recoverable from
    # this data at all, even with the mean-distance backup. Must degrade
    # gracefully (stopped=False, a distinctly labeled stop_reason), never
    # raise and crash the caller's run -- tau still reflects the pulls
    # actually made, since (unlike kabc's pilot) those pulls are the same
    # ones clustering would have used.
    env = _FullyDegenerateEnv(N=12, K=3)
    N = env.n_arms
    budget_cap = 1200
    n = budget_cap // N
    rng = np.random.default_rng(0)

    with pytest.warns(UserWarning, match="only the bandwidth/clustering step failed"):
        result = fit(env, delta=0.05, budget_cap=budget_cap, rng=rng)

    assert result.stopped is False
    assert result.history == [{"stop_reason": STOP_REASON_DEGENERATE_BANDWIDTH}]
    assert result.tau == N * n
    assert result.partition.shape == (N,)
    assert np.all(result.partition == result.partition[0])
