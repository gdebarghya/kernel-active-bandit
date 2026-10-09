import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from kabc.algorithms.kabc import N_PILOT, fit_oracle
from kabc.environments import GaussianMixtureEnv
from kabc.kernels import median_heuristic_bandwidth, mmd2_plugin, rkhs_variance
from kabc.thresholds import oracle_budget


def _true_s_star_sq(env, rng, n_oracle_sample=3000):
    """Estimate the instance's TRUE s_star_sq (Eq. (5)) via a large auxiliary
    per-cluster sample -- NOT counted toward any tau, and not part of the
    library's public interface (fit_oracle intentionally does not compute
    this itself; see kabc.py's fit_oracle docstring). This mirrors what an
    experiment script would precompute offline for a known synthetic
    instance, using the same bandwidth-then-sample recipe fit_oracle uses
    internally (one pilot, then per-cluster samples), so it's consistent
    with what fit_oracle will actually see.
    """
    pilot = [
        np.stack([np.atleast_1d(env.pull(i, rng)) for _ in range(N_PILOT)]) for i in range(env.n_arms)
    ]
    bandwidth = median_heuristic_bandwidth(np.concatenate(pilot, axis=0))

    cluster_samples = []
    for k in range(env.K):
        arm = int(np.where(env.true_partition == k)[0][0])
        cluster_samples.append(
            np.stack([np.atleast_1d(env.pull(arm, rng)) for _ in range(n_oracle_sample)])
        )

    g_bar = 1.0  # RBF kernel's sup
    s_star_sq = np.inf
    for i in range(env.K):
        for j in range(env.K):
            if i == j:
                continue
            mmd2 = mmd2_plugin(cluster_samples[i], cluster_samples[j], bandwidth)
            v_i = rkhs_variance(cluster_samples[i], bandwidth)
            v_j = rkhs_variance(cluster_samples[j], bandwidth)
            gap_term = mmd2 / max(v_i, v_j)
            kernel_term = 2.0 * np.sqrt(mmd2) / np.sqrt(g_bar)
            s_star_sq = min(s_star_sq, gap_term, kernel_term)
    return s_star_sq


# --------------------------------------------------------------------------
# correct recovery on an easy instance with correctly-computed s_star_sq
# --------------------------------------------------------------------------

def test_recovers_partition_with_true_s_star_sq():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    s_star_sq = _true_s_star_sq(env, np.random.default_rng(42))

    rng = np.random.default_rng(0)
    result = fit_oracle(env, delta=0.05, budget_cap=1_000_000, rng=rng, s_star_sq=s_star_sq)

    assert result.stopped is True
    assert adjusted_rand_score(env.true_partition, result.partition) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# tau matches oracle_budget's formula plus the pilot cost, exactly
# --------------------------------------------------------------------------

def test_tau_is_pilot_plus_single_shot_oracle_budget():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    N = env.n_arms
    s_star_sq = _true_s_star_sq(env, np.random.default_rng(42))

    n_star = oracle_budget(s_star_sq, N, 0.05)
    expected_tau = N * N_PILOT + N * n_star

    rng = np.random.default_rng(0)
    result = fit_oracle(env, delta=0.05, budget_cap=expected_tau + 1, rng=rng, s_star_sq=s_star_sq)

    assert result.tau == expected_tau
    # single-shot: exactly one CLUSTER call, no doubling-trick iteration overhead
    assert len(result.history) == 1
    assert result.history[0]["k"] == 1
    assert result.history[0]["n_k"] == n_star
    assert result.history[0]["delta_k"] == 0.05  # delta itself, not a decayed delta_k
