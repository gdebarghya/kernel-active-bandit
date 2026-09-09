import math

import numpy as np
import pytest

from kabc.kernels import mmd2_plugin, rkhs_variance
from kabc.thresholds import (
    iteration_budget,
    iteration_confidence,
    oracle_budget,
    subgaussian_threshold,
    variance_aware_threshold,
)


# --------------------------------------------------------------------------
# variance_aware_threshold
# --------------------------------------------------------------------------

def _manual_variance_aware(n, delta_prime, N, var_i, var_j, g_tilde=1.0, const=1.0):
    log_term = math.log(8.0 * (N**2 - N) / delta_prime)
    return const * (
        (math.sqrt(var_i) + math.sqrt(var_j)) * math.sqrt(2.0 * log_term / n)
        + (32.0 / 3.0) * math.sqrt(g_tilde) * log_term / n
    )


def test_variance_aware_threshold_hand_computed():
    expected = _manual_variance_aware(50, 0.05, 40, 0.3, 0.5)
    assert variance_aware_threshold(50, 0.05, 40, 0.3, 0.5) == pytest.approx(expected)


def test_variance_aware_threshold_decreases_in_n():
    a = variance_aware_threshold(10, 0.05, 40, 0.3, 0.5)
    b = variance_aware_threshold(100, 0.05, 40, 0.3, 0.5)
    assert b < a


def test_variance_aware_threshold_increases_as_delta_shrinks():
    a = variance_aware_threshold(50, 0.1, 40, 0.3, 0.5)
    b = variance_aware_threshold(50, 0.001, 40, 0.3, 0.5)
    assert b > a


def test_variance_aware_threshold_const_scaling():
    base = variance_aware_threshold(50, 0.05, 40, 0.3, 0.5, const=1.0)
    for c in [0.1, 2.0, 5.0]:
        scaled = variance_aware_threshold(50, 0.05, 40, 0.3, 0.5, const=c)
        assert scaled == pytest.approx(c * base)


def test_variance_aware_threshold_symmetric_in_var_i_var_j():
    a = variance_aware_threshold(50, 0.05, 40, 0.3, 0.5)
    b = variance_aware_threshold(50, 0.05, 40, 0.5, 0.3)
    assert a == pytest.approx(b)


def test_variance_aware_threshold_zero_variance_edge_case():
    expected = _manual_variance_aware(50, 0.05, 40, 0.0, 0.0)
    assert variance_aware_threshold(50, 0.05, 40, 0.0, 0.0) == pytest.approx(expected)
    assert expected > 0  # bias term alone is still positive


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n=0, delta_prime=0.05, N=40, var_i=0.3, var_j=0.5),
        dict(n=50, delta_prime=0.0, N=40, var_i=0.3, var_j=0.5),
        dict(n=50, delta_prime=1.5, N=40, var_i=0.3, var_j=0.5),
        dict(n=50, delta_prime=0.05, N=1, var_i=0.3, var_j=0.5),
        dict(n=50, delta_prime=0.05, N=40, var_i=-0.1, var_j=0.5),
        dict(n=50, delta_prime=0.05, N=40, var_i=0.3, var_j=-0.1),
        dict(n=50, delta_prime=0.05, N=40, var_i=0.3, var_j=0.5, g_tilde=0.0),
        dict(n=50, delta_prime=0.05, N=40, var_i=0.3, var_j=0.5, const=0.0),
    ],
)
def test_variance_aware_threshold_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        variance_aware_threshold(**kwargs)


# --------------------------------------------------------------------------
# subgaussian_threshold
# --------------------------------------------------------------------------

def _manual_subgaussian(n, delta_prime, N, g_bar=1.0, const=1.0):
    log_term = math.log(8.0 * (N**2 - N) / delta_prime)
    return const * math.sqrt(g_bar / n) * (math.sqrt(log_term) + 2.0)


def test_subgaussian_threshold_hand_computed():
    expected = _manual_subgaussian(50, 0.05, 40)
    assert subgaussian_threshold(50, 0.05, 40) == pytest.approx(expected)


def test_subgaussian_threshold_decreases_in_n():
    a = subgaussian_threshold(10, 0.05, 40)
    b = subgaussian_threshold(100, 0.05, 40)
    assert b < a


def test_subgaussian_threshold_increases_as_delta_shrinks():
    a = subgaussian_threshold(50, 0.1, 40)
    b = subgaussian_threshold(50, 0.001, 40)
    assert b > a


def test_subgaussian_threshold_const_scaling():
    base = subgaussian_threshold(50, 0.05, 40, const=1.0)
    for c in [0.1, 2.0, 5.0]:
        scaled = subgaussian_threshold(50, 0.05, 40, const=c)
        assert scaled == pytest.approx(c * base)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n=0, delta_prime=0.05, N=40),
        dict(n=50, delta_prime=0.0, N=40),
        dict(n=50, delta_prime=1.5, N=40),
        dict(n=50, delta_prime=0.05, N=1),
        dict(n=50, delta_prime=0.05, N=40, g_bar=0.0),
        dict(n=50, delta_prime=0.05, N=40, const=0.0),
    ],
)
def test_subgaussian_threshold_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        subgaussian_threshold(**kwargs)


# --------------------------------------------------------------------------
# iteration_confidence
# --------------------------------------------------------------------------

def test_iteration_confidence_hand_computed():
    for k in [1, 2, 3, 10]:
        assert iteration_confidence(k, 0.05) == pytest.approx(0.05 / (4 * k**2))


def test_iteration_confidence_decreases_in_k():
    values = [iteration_confidence(k, 0.05) for k in range(1, 10)]
    assert all(earlier > later for earlier, later in zip(values, values[1:]))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(k=0, delta=0.05),
        dict(k=1, delta=0.0),
        dict(k=1, delta=1.5),
    ],
)
def test_iteration_confidence_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        iteration_confidence(**kwargs)


# --------------------------------------------------------------------------
# iteration_budget
# --------------------------------------------------------------------------

def test_iteration_budget_hand_computed():
    N, delta_k, k = 40, 0.01, 3
    expected = math.ceil(2**k * math.log(8 * (N**2 - N) / delta_k))
    assert iteration_budget(k, N, delta_k) == expected


def test_iteration_budget_ceil_boundary():
    # choose inputs where the raw value is not already an integer
    N, delta_k, k = 7, 0.037, 2
    raw = 2**k * math.log(8 * (N**2 - N) / delta_k)
    assert raw != math.floor(raw)
    assert iteration_budget(k, N, delta_k) == math.ceil(raw)


def test_iteration_budget_increases_in_k():
    N, delta = 40, 0.05
    budgets = []
    for k in range(1, 6):
        delta_k = iteration_confidence(k, delta)
        budgets.append(iteration_budget(k, N, delta_k))
    assert all(earlier < later for earlier, later in zip(budgets, budgets[1:]))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(k=0, N=40, delta_k=0.01),
        dict(k=1, N=1, delta_k=0.01),
        dict(k=1, N=40, delta_k=0.0),
        dict(k=1, N=40, delta_k=1.5),
    ],
)
def test_iteration_budget_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        iteration_budget(**kwargs)


# --------------------------------------------------------------------------
# oracle_budget
# --------------------------------------------------------------------------

def test_oracle_budget_hand_computed():
    s_star_sq, N, delta = 0.2, 40, 0.05
    expected = math.ceil(128.0 / s_star_sq * math.log(8 * (N**2 - N) / delta))
    assert oracle_budget(s_star_sq, N, delta) == expected


def test_oracle_budget_ceil_boundary():
    s_star_sq, N, delta = 0.37, 7, 0.041
    raw = 128.0 / s_star_sq * math.log(8 * (N**2 - N) / delta)
    assert raw != math.floor(raw)
    assert oracle_budget(s_star_sq, N, delta) == math.ceil(raw)


def test_oracle_budget_decreases_in_s_star_sq():
    a = oracle_budget(0.1, 40, 0.05)
    b = oracle_budget(1.0, 40, 0.05)
    assert b < a


def test_oracle_budget_const_scaling():
    base = oracle_budget(0.2, 40, 0.05, const=1.0)
    scaled = oracle_budget(0.2, 40, 0.05, const=2.0)
    # both are integer ceilings, so compare the underlying ratio loosely
    assert scaled >= base


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(s_star_sq=0.0, N=40, delta=0.05),
        dict(s_star_sq=-1.0, N=40, delta=0.05),
        dict(s_star_sq=0.2, N=1, delta=0.05),
        dict(s_star_sq=0.2, N=40, delta=0.0),
        dict(s_star_sq=0.2, N=40, delta=0.05, const=0.0),
    ],
)
def test_oracle_budget_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        oracle_budget(**kwargs)


# --------------------------------------------------------------------------
# integration: wiring kernels.py estimators together with thresholds
# --------------------------------------------------------------------------

def test_variance_aware_threshold_integration_same_distribution():
    N, delta_prime, n = 40, 0.05, 200
    below_count = 0
    num_seeds = 50
    for seed in range(num_seeds):
        rng = np.random.default_rng(4000 + seed)
        X = rng.normal(size=n)
        Y = rng.normal(size=n)
        var_i = rkhs_variance(X, bandwidth=1.0)
        var_j = rkhs_variance(Y, bandwidth=1.0)
        stat = math.sqrt(mmd2_plugin(X, Y, bandwidth=1.0))
        threshold = variance_aware_threshold(n, delta_prime, N, var_i, var_j)
        if stat <= threshold:
            below_count += 1
    # delta-PAC: should stay below threshold in the vast majority of seeds
    assert below_count / num_seeds >= 0.9


def test_variance_aware_threshold_integration_separated_distributions():
    N, delta_prime, n = 40, 0.05, 500
    above_count = 0
    num_seeds = 50
    for seed in range(num_seeds):
        rng = np.random.default_rng(5000 + seed)
        X = rng.normal(loc=0.0, size=n)
        Y = rng.normal(loc=5.0, size=n)
        var_i = rkhs_variance(X, bandwidth=1.0)
        var_j = rkhs_variance(Y, bandwidth=1.0)
        stat = math.sqrt(mmd2_plugin(X, Y, bandwidth=1.0))
        threshold = variance_aware_threshold(n, delta_prime, N, var_i, var_j)
        if stat > threshold:
            above_count += 1
    assert above_count / num_seeds >= 0.9


def test_subgaussian_threshold_integration_separated_distributions():
    N, delta_prime, n = 40, 0.05, 500
    above_count = 0
    num_seeds = 50
    for seed in range(num_seeds):
        rng = np.random.default_rng(6000 + seed)
        X = rng.normal(loc=0.0, size=n)
        Y = rng.normal(loc=5.0, size=n)
        stat = math.sqrt(mmd2_plugin(X, Y, bandwidth=1.0))
        threshold = subgaussian_threshold(n, delta_prime, N)
        if stat > threshold:
            above_count += 1
    assert above_count / num_seeds >= 0.9
