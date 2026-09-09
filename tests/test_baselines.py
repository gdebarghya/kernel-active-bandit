import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from kabc.algorithms.baselines import fit
from kabc.environments import GaussianMixtureEnv, ScalePairEnv


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
