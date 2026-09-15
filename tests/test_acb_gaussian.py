import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from kabc.algorithms.acb_gaussian import fit
from kabc.environments import GaussianMixtureEnv, MomentMatchedMixtureEnv, ScalePairEnv


# --------------------------------------------------------------------------
# sanity check: ACB-Gaussian recovers the partition, with no calibrated
# constant anywhere in its formulas (see module docstring's CALIBRATION note)
# --------------------------------------------------------------------------

def test_recovers_partition_on_well_separated_gaussian_instance():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    rng = np.random.default_rng(0)
    result = fit(env, delta=0.05, budget_cap=200_000, rng=rng)

    assert result.stopped is True
    assert adjusted_rand_score(env.true_partition, result.partition) == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_recovers_partition_across_seeds_no_calibration(seed):
    """Repeats the well-separated case across independent seeds -- there is
    no c_cal-style constant anywhere in acb_gaussian.py to tune, so this
    should pass reliably on a well-separated instance for any seed, not
    just a cherry-picked one.
    """
    env = GaussianMixtureEnv(N=15, K=3, d=10, Delta=6.0)
    rng = np.random.default_rng(seed)
    result = fit(env, delta=0.05, budget_cap=200_000, rng=rng)

    assert result.stopped is True
    assert adjusted_rand_score(env.true_partition, result.partition) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the safeguard: acb-gaussian must never silently run on a non-Gaussian
# environment (its formulas rely on exactly-known Gaussian noise) -- it must
# hard-fail immediately, not return a meaningless or misleading result
# --------------------------------------------------------------------------

def test_rejects_scale_pair_env():
    env = ScalePairEnv(N=40, sigma_B=3.0)
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="GaussianMixtureEnv"):
        fit(env, delta=0.05, budget_cap=50_000, rng=rng)


def test_rejects_moment_matched_mixture_env():
    env = MomentMatchedMixtureEnv(N=40, a=0.9)
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="GaussianMixtureEnv"):
        fit(env, delta=0.05, budget_cap=50_000, rng=rng)


# --------------------------------------------------------------------------
# input validation, mirroring acb.py's own conventions
# --------------------------------------------------------------------------

def test_rejects_invalid_delta():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="delta"):
        fit(env, delta=0.0, budget_cap=1000, rng=rng)


def test_rejects_invalid_budget_cap():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="budget_cap"):
        fit(env, delta=0.05, budget_cap=0, rng=rng)


# --------------------------------------------------------------------------
# graceful behavior under a too-tight budget (no crash, stopped=False)
# --------------------------------------------------------------------------

def test_graceful_stop_under_tiny_budget():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    rng = np.random.default_rng(0)
    result = fit(env, delta=0.05, budget_cap=1, rng=rng)

    assert result.stopped is False
    assert result.tau <= 1
