import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from kabc.algorithms.acb import fit
from kabc.environments import GaussianMixtureEnv, ScalePairEnv


# --------------------------------------------------------------------------
# sanity check: ACB works at all, on a well-separated Gaussian instance
# --------------------------------------------------------------------------

def test_recovers_partition_on_well_separated_gaussian_instance():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    rng = np.random.default_rng(0)
    result = fit(env, delta=0.05, budget_cap=200_000, rng=rng)

    assert result.stopped is True
    assert adjusted_rand_score(env.true_partition, result.partition) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the property Experiment 2 depends on: ACB is mean-blind, so on a same-mean
# different-variance instance it must NOT separate the clusters
# --------------------------------------------------------------------------

@pytest.mark.parametrize("budget_cap", [5_000, 50_000])
def test_does_not_separate_same_mean_different_variance_instance(budget_cap):
    env = ScalePairEnv(N=40, sigma_B=3.0)
    rng = np.random.default_rng(1)
    result = fit(env, delta=0.05, budget_cap=budget_cap, rng=rng)

    # ACB only ever looks at estimated means, which are equal (both clusters
    # are centered at 0) -- it must fail to find K=2 representatives and never
    # stop, regardless of budget, and its best-effort partition must carry no
    # real information about the true (variance-based) clustering.
    assert result.stopped is False
    assert adjusted_rand_score(env.true_partition, result.partition) == pytest.approx(0.0)
