import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from kabc.algorithms.kabc import N_PILOT, fit, fit_sg
from kabc.environments import GaussianMixtureEnv
from kabc.thresholds import iteration_budget, iteration_confidence


# --------------------------------------------------------------------------
# correct recovery on an easy, well-separated instance
# --------------------------------------------------------------------------

@pytest.mark.parametrize("algorithm", [fit, fit_sg])
def test_recovers_partition_on_well_separated_instance(algorithm):
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    rng = np.random.default_rng(0)
    result = algorithm(env, delta=0.05, budget_cap=50_000, rng=rng)
    assert result.stopped is True
    assert adjusted_rand_score(env.true_partition, result.partition) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# budget scaling matches the theoretical n_k schedule
# --------------------------------------------------------------------------

def test_history_matches_iteration_budget_schedule():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    rng = np.random.default_rng(0)
    result = fit(env, delta=0.05, budget_cap=50_000, rng=rng)

    assert len(result.history) >= 1
    for entry in result.history:
        k = entry["k"]
        expected_delta_k = iteration_confidence(k, 0.05)
        expected_n_k = iteration_budget(k, env.n_arms, expected_delta_k)
        assert entry["delta_k"] == expected_delta_k
        assert entry["n_k"] == expected_n_k

    # tau accounts for the pilot phase plus every completed iteration's fresh
    # per-arm sample (no cumulative reuse across iterations, matching
    # Theorem 3.1's budget bound tau <= sum_k N*n_k).
    expected_tau = env.n_arms * N_PILOT + sum(env.n_arms * entry["n_k"] for entry in result.history)
    assert result.tau == expected_tau


def test_iteration_ks_are_consecutive_starting_at_one():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    rng = np.random.default_rng(0)
    result = fit(env, delta=0.05, budget_cap=50_000, rng=rng)
    ks = [entry["k"] for entry in result.history]
    assert ks == list(range(1, len(ks) + 1))


# --------------------------------------------------------------------------
# behavior at budget_cap: best-effort, no crash
# --------------------------------------------------------------------------

def test_stops_gracefully_when_cap_forbids_even_first_iteration():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    N = env.n_arms
    pilot_cost = N * N_PILOT
    delta_1 = iteration_confidence(1, 0.05)
    n_1 = iteration_budget(1, N, delta_1)
    cost_1 = N * n_1
    tiny_cap = pilot_cost + cost_1 - 1  # affords the pilot, not iteration k=1

    rng = np.random.default_rng(0)
    result = fit(env, delta=0.05, budget_cap=tiny_cap, rng=rng)

    assert result.stopped is False
    assert result.history == []
    assert result.tau == pilot_cost
    assert result.partition.shape == (N,)
    # best-effort fallback: no iteration ran, so all arms are reported as one cluster
    assert np.all(result.partition == result.partition[0])


def test_stops_gracefully_after_partial_progress():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    N = env.n_arms
    pilot_cost = N * N_PILOT
    delta_1 = iteration_confidence(1, 0.05)
    n_1 = iteration_budget(1, N, delta_1)
    delta_2 = iteration_confidence(2, 0.05)
    n_2 = iteration_budget(2, N, delta_2)
    cap_after_one_iteration = pilot_cost + N * n_1 + N * n_2 - 1  # affords k=1, not k=2

    rng = np.random.default_rng(0)
    result = fit(env, delta=0.05, budget_cap=cap_after_one_iteration, rng=rng)

    assert result.stopped is False
    assert len(result.history) == 1
    assert result.history[0]["k"] == 1
    assert result.tau == pilot_cost + N * n_1
    assert result.partition.shape == (N,)
    assert result.partition.dtype.kind in "iu"


def test_warns_and_returns_gracefully_when_cap_cannot_afford_pilot_phase():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    N = env.n_arms
    pilot_cost = N * N_PILOT
    rng = np.random.default_rng(0)

    with pytest.warns(UserWarning):
        result = fit(env, delta=0.05, budget_cap=pilot_cost - 1, rng=rng)

    assert result.stopped is False
    assert result.tau == 0
    assert result.history == []
    assert result.partition.shape == (N,)
    assert np.all(result.partition == result.partition[0])
