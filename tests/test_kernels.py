import numpy as np
import pytest

from kabc.kernels import (
    mmd2_and_variance,
    mmd2_plugin,
    mmd2_unbiased,
    median_heuristic_bandwidth,
    rbf_kernel,
    rkhs_variance,
)


# --------------------------------------------------------------------------
# rbf_kernel
# --------------------------------------------------------------------------

def test_rbf_kernel_hand_computed_1d():
    x = np.array([0.0])
    y = np.array([1.0])
    K = rbf_kernel(x.reshape(1, 1), y.reshape(1, 1), bandwidth=1.0)
    assert K.shape == (1, 1)
    assert K[0, 0] == pytest.approx(np.exp(-0.5))


def test_rbf_kernel_diagonal_is_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    K = rbf_kernel(X, bandwidth=2.0)
    np.testing.assert_allclose(np.diag(K), 1.0)


def test_rbf_kernel_symmetric():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 2))
    K = rbf_kernel(X, bandwidth=1.5)
    np.testing.assert_allclose(K, K.T)


def test_rbf_kernel_1d_2d_equivalence():
    rng = np.random.default_rng(2)
    x = rng.normal(size=6)
    K_1d = rbf_kernel(x, bandwidth=1.0)
    K_2d = rbf_kernel(x.reshape(-1, 1), bandwidth=1.0)
    np.testing.assert_allclose(K_1d, K_2d)


@pytest.mark.parametrize("c", [0.5, 2.0, 10.0])
def test_rbf_kernel_scale_invariant(c):
    # k(x,y) = exp(-||x-y||^2 / (2*bandwidth**2)) depends only on ||x-y||/bandwidth,
    # so jointly rescaling the data and the bandwidth by c must leave K unchanged.
    rng = np.random.default_rng(5)
    X = rng.normal(size=(7, 3))
    bandwidth = 1.4
    K = rbf_kernel(X, bandwidth=bandwidth)
    K_scaled = rbf_kernel(c * X, bandwidth=c * bandwidth)
    np.testing.assert_allclose(K_scaled, K)


def test_rbf_kernel_larger_bandwidth_increases_offdiag_values():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(6, 2))
    K_small = rbf_kernel(X, bandwidth=0.5)
    K_large = rbf_kernel(X, bandwidth=5.0)
    off_diag = ~np.eye(X.shape[0], dtype=bool)
    assert np.all(K_large[off_diag] >= K_small[off_diag])


@pytest.mark.parametrize("bandwidth", [0.0, -1.0])
def test_rbf_kernel_rejects_nonpositive_bandwidth(bandwidth):
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError):
        rbf_kernel(X, bandwidth=bandwidth)


def test_rbf_kernel_rejects_dimension_mismatch():
    X = np.zeros((3, 3))
    Y = np.zeros((3, 5))
    with pytest.raises(ValueError):
        rbf_kernel(X, Y, bandwidth=1.0)


# --------------------------------------------------------------------------
# median_heuristic_bandwidth
# --------------------------------------------------------------------------

def test_median_heuristic_known_value():
    X = np.array([0.0, 1.0, 2.0, 3.0])
    # pairwise distances: 1,2,3,1,2,1 -> sorted [1,1,1,2,2,3] -> median 1.5
    assert median_heuristic_bandwidth(X) == pytest.approx(1.5)


def test_median_heuristic_matches_bruteforce():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(15, 2))
    n = X.shape[0]
    distances = [
        np.linalg.norm(X[i] - X[j])
        for i in range(n)
        for j in range(n)
        if i < j
    ]
    expected = float(np.median(distances))
    assert median_heuristic_bandwidth(X) == pytest.approx(expected)


def test_median_heuristic_1d_2d_equivalence():
    rng = np.random.default_rng(4)
    x = rng.normal(size=9)
    assert median_heuristic_bandwidth(x) == pytest.approx(
        median_heuristic_bandwidth(x.reshape(-1, 1))
    )


def test_median_heuristic_degenerate_raises():
    X = np.ones((5, 2))
    with pytest.raises(ValueError):
        median_heuristic_bandwidth(X)


def test_median_heuristic_single_point_raises():
    X = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        median_heuristic_bandwidth(X)


# --------------------------------------------------------------------------
# mmd2_unbiased
# --------------------------------------------------------------------------

def _bruteforce_mmd2(X, Y, bandwidth):
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    Y = np.atleast_2d(Y.T).T if Y.ndim == 1 else Y
    n_x, n_y = X.shape[0], Y.shape[0]

    def k(a, b):
        return np.exp(-np.sum((a - b) ** 2) / (2 * bandwidth**2))

    term_x = sum(k(X[i], X[j]) for i in range(n_x) for j in range(n_x) if i != j)
    term_x /= n_x * (n_x - 1)

    term_y = sum(k(Y[i], Y[j]) for i in range(n_y) for j in range(n_y) if i != j)
    term_y /= n_y * (n_y - 1)

    term_xy = sum(k(X[i], Y[j]) for i in range(n_x) for j in range(n_y))
    term_xy /= n_x * n_y

    return term_x + term_y - 2 * term_xy


@pytest.mark.parametrize("n_x,n_y,dim", [(5, 7, 1), (6, 6, 1), (4, 9, 3)])
def test_mmd2_unbiased_matches_bruteforce_reference(n_x, n_y, dim):
    rng = np.random.default_rng(10 + n_x + n_y + dim)
    X = rng.normal(size=(n_x, dim)) if dim > 1 else rng.normal(size=n_x)
    Y = rng.normal(loc=0.5, size=(n_y, dim)) if dim > 1 else rng.normal(loc=0.5, size=n_y)
    bandwidth = 1.3
    got = mmd2_unbiased(X, Y, bandwidth)
    expected = _bruteforce_mmd2(np.asarray(X, dtype=float), np.asarray(Y, dtype=float), bandwidth)
    assert got == pytest.approx(expected, rel=1e-10)


def test_mmd2_unbiased_symmetric_in_XY():
    rng = np.random.default_rng(20)
    X = rng.normal(size=8)
    Y = rng.normal(loc=1.0, size=11)
    assert mmd2_unbiased(X, Y, bandwidth=1.0) == pytest.approx(
        mmd2_unbiased(Y, X, bandwidth=1.0)
    )


def _closed_form_gaussian_mmd2(mu1, s1, mu2, s2, sigma):
    def e_exp(m, v):
        # E[exp(-Z^2 / (2 sigma^2))] for Z ~ N(m, v)
        return sigma / np.sqrt(sigma**2 + v) * np.exp(-(m**2) / (2 * (sigma**2 + v)))

    e_xx = e_exp(0.0, 2 * s1**2)
    e_yy = e_exp(0.0, 2 * s2**2)
    e_xy = e_exp(mu1 - mu2, s1**2 + s2**2)
    return e_xx + e_yy - 2 * e_xy


def test_mmd2_unbiased_matches_closed_form_gaussian():
    mu1, s1 = 0.0, 1.0
    mu2, s2 = 1.0, 2.0
    sigma = 1.5
    n = 4000
    num_seeds = 5

    estimates = []
    variances = []
    for seed in range(num_seeds):
        rng = np.random.default_rng(100 + seed)
        X = rng.normal(mu1, s1, size=n)
        Y = rng.normal(mu2, s2, size=n)
        mmd2, var = mmd2_and_variance(X, Y, sigma)
        estimates.append(mmd2)
        variances.append(var)

    mean_estimate = np.mean(estimates)
    # standard error of the mean over `num_seeds` independent estimates
    se = np.sqrt(np.mean(variances) / num_seeds)
    expected = _closed_form_gaussian_mmd2(mu1, s1, mu2, s2, sigma)

    assert mean_estimate == pytest.approx(expected, abs=8 * se)


def test_mmd2_unbiased_centered_near_zero_under_null():
    n = 500
    num_reps = 200
    estimates = []
    for seed in range(num_reps):
        rng = np.random.default_rng(1000 + seed)
        X = rng.normal(size=n)
        Y = rng.normal(size=n)
        estimates.append(mmd2_unbiased(X, Y, bandwidth=1.0))

    estimates = np.array(estimates)
    # under H0 the estimator is unbiased for MMD^2=0 but can be negative
    assert np.any(estimates < 0)
    se = np.std(estimates, ddof=1) / np.sqrt(num_reps)
    assert np.mean(estimates) == pytest.approx(0.0, abs=6 * se)


@pytest.mark.parametrize("n_x,n_y", [(1, 5), (5, 1), (1, 1)])
def test_mmd2_unbiased_rejects_small_samples(n_x, n_y):
    X = np.zeros(n_x)
    Y = np.zeros(n_y)
    with pytest.raises(ValueError):
        mmd2_unbiased(X, Y, bandwidth=1.0)


def test_mmd2_unbiased_rejects_dimension_mismatch():
    X = np.zeros((4, 2))
    Y = np.zeros((4, 3))
    with pytest.raises(ValueError):
        mmd2_unbiased(X, Y, bandwidth=1.0)


# --------------------------------------------------------------------------
# mmd2_and_variance
# --------------------------------------------------------------------------

def test_mmd2_and_variance_matches_mmd2_unbiased():
    rng = np.random.default_rng(30)
    X = rng.normal(size=12)
    Y = rng.normal(loc=0.3, size=9)
    mmd2, _ = mmd2_and_variance(X, Y, bandwidth=1.0)
    assert mmd2 == pytest.approx(mmd2_unbiased(X, Y, bandwidth=1.0))


@pytest.mark.parametrize("seed", range(5))
def test_mmd2_and_variance_variance_is_nonnegative(seed):
    rng = np.random.default_rng(40 + seed)
    X = rng.normal(size=20)
    Y = rng.normal(loc=0.2, size=15)
    _, var = mmd2_and_variance(X, Y, bandwidth=1.2)
    assert var >= 0.0


def test_mmd2_and_variance_handles_minimum_sample_size():
    X = np.array([0.0, 1.0])
    Y = np.array([2.0, 3.0])
    mmd2, var = mmd2_and_variance(X, Y, bandwidth=1.0)
    assert np.isfinite(mmd2)
    assert np.isfinite(var)
    assert var >= 0.0


def test_mmd2_and_variance_monte_carlo_matches_empirical_variance():
    # Exp-2a scale pair: same mean, different variance.
    n = 60
    num_reps = 400
    sigma = 1.0

    mmd2_estimates = []
    variance_estimates = []
    for seed in range(num_reps):
        rng = np.random.default_rng(2000 + seed)
        X = rng.normal(0.0, 1.0, size=n)
        Y = rng.normal(0.0, 2.0, size=n)
        mmd2, var = mmd2_and_variance(X, Y, sigma)
        mmd2_estimates.append(mmd2)
        variance_estimates.append(var)

    empirical_variance = np.var(mmd2_estimates, ddof=1)
    predicted_variance = np.mean(variance_estimates)

    # The variance estimate is an asymptotic (first-order) approximation, so
    # exact agreement isn't expected -- check it's within a generous factor.
    assert predicted_variance / 3 <= empirical_variance <= predicted_variance * 3


@pytest.mark.parametrize("n_x,n_y", [(1, 5), (5, 1)])
def test_mmd2_and_variance_rejects_small_samples(n_x, n_y):
    X = np.zeros(n_x)
    Y = np.zeros(n_y)
    with pytest.raises(ValueError):
        mmd2_and_variance(X, Y, bandwidth=1.0)


# --------------------------------------------------------------------------
# rkhs_variance
# --------------------------------------------------------------------------

def _bruteforce_rkhs_variance(X, bandwidth):
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    n = X.shape[0]

    def k(a, b):
        return np.exp(-np.sum((a - b) ** 2) / (2 * bandwidth**2))

    total = 0.0
    for t in range(n):
        row_mean = sum(k(X[t], X[s]) for s in range(n)) / n
        total += k(X[t], X[t]) - row_mean
    return total / (n - 1)


def test_rkhs_variance_hand_computed():
    X = np.array([0.0, 1.0, 3.0])
    expected = _bruteforce_rkhs_variance(X, bandwidth=1.0)
    assert rkhs_variance(X, bandwidth=1.0) == pytest.approx(expected)


def test_rkhs_variance_matches_bruteforce():
    rng = np.random.default_rng(50)
    X = rng.normal(size=(9, 2))
    expected = _bruteforce_rkhs_variance(X, bandwidth=1.4)
    assert rkhs_variance(X, bandwidth=1.4) == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(10))
def test_rkhs_variance_nonnegative(seed):
    rng = np.random.default_rng(1000 + seed)
    n = rng.integers(2, 20)
    d = rng.integers(1, 4)
    X = rng.normal(size=(n, d))
    assert rkhs_variance(X, bandwidth=1.0) >= 0.0


def test_rkhs_variance_1d_2d_equivalence():
    rng = np.random.default_rng(51)
    x = rng.normal(size=7)
    assert rkhs_variance(x, bandwidth=1.0) == pytest.approx(
        rkhs_variance(x.reshape(-1, 1), bandwidth=1.0)
    )


def test_rkhs_variance_single_point_raises():
    X = np.array([1.0])
    with pytest.raises(ValueError):
        rkhs_variance(X, bandwidth=1.0)


@pytest.mark.parametrize("bandwidth", [0.0, -1.0])
def test_rkhs_variance_rejects_nonpositive_bandwidth(bandwidth):
    X = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        rkhs_variance(X, bandwidth=bandwidth)


# --------------------------------------------------------------------------
# mmd2_plugin
# --------------------------------------------------------------------------

def _bruteforce_mmd2_plugin(X, Y, bandwidth):
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    Y = np.atleast_2d(Y.T).T if Y.ndim == 1 else Y
    n_x, n_y = X.shape[0], Y.shape[0]

    def k(a, b):
        return np.exp(-np.sum((a - b) ** 2) / (2 * bandwidth**2))

    term_x = sum(k(X[s], X[t]) for s in range(n_x) for t in range(n_x)) / n_x**2
    term_y = sum(k(Y[s], Y[t]) for s in range(n_y) for t in range(n_y)) / n_y**2
    term_xy = sum(k(X[s], Y[t]) for s in range(n_x) for t in range(n_y)) / (n_x * n_y)
    return term_x + term_y - 2 * term_xy


def test_mmd2_plugin_exactly_zero_for_identical_sample():
    rng = np.random.default_rng(60)
    X = rng.normal(size=5)
    assert mmd2_plugin(X, X, bandwidth=1.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_mmd2_plugin_nonnegative(seed):
    rng = np.random.default_rng(2000 + seed)
    n_x, n_y = rng.integers(1, 10), rng.integers(1, 10)
    X = rng.normal(size=n_x)
    Y = rng.normal(loc=0.5, size=n_y)
    assert mmd2_plugin(X, Y, bandwidth=1.0) >= 0.0


def test_mmd2_plugin_hand_computed():
    X = np.array([0.0, 1.0])
    Y = np.array([2.0, 3.0])
    expected = _bruteforce_mmd2_plugin(X, Y, bandwidth=1.0)
    assert mmd2_plugin(X, Y, bandwidth=1.0) == pytest.approx(expected)


@pytest.mark.parametrize("n_x,n_y,dim", [(5, 5, 1), (6, 6, 3)])
def test_mmd2_plugin_matches_bruteforce_equal_n(n_x, n_y, dim):
    rng = np.random.default_rng(70 + n_x + dim)
    X = rng.normal(size=(n_x, dim)) if dim > 1 else rng.normal(size=n_x)
    Y = rng.normal(loc=0.4, size=(n_y, dim)) if dim > 1 else rng.normal(loc=0.4, size=n_y)
    bandwidth = 1.2
    expected = _bruteforce_mmd2_plugin(
        np.asarray(X, dtype=float), np.asarray(Y, dtype=float), bandwidth
    )
    assert mmd2_plugin(X, Y, bandwidth) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("n_x,n_y,dim", [(5, 8, 1), (4, 9, 2)])
def test_mmd2_plugin_matches_bruteforce_unequal_n(n_x, n_y, dim):
    rng = np.random.default_rng(80 + n_x + n_y + dim)
    X = rng.normal(size=(n_x, dim)) if dim > 1 else rng.normal(size=n_x)
    Y = rng.normal(loc=0.4, size=(n_y, dim)) if dim > 1 else rng.normal(loc=0.4, size=n_y)
    bandwidth = 0.9
    expected = _bruteforce_mmd2_plugin(
        np.asarray(X, dtype=float), np.asarray(Y, dtype=float), bandwidth
    )
    assert mmd2_plugin(X, Y, bandwidth) == pytest.approx(expected, rel=1e-10)


def test_mmd2_plugin_symmetric_in_XY():
    rng = np.random.default_rng(90)
    X = rng.normal(size=7)
    Y = rng.normal(loc=0.6, size=10)
    assert mmd2_plugin(X, Y, bandwidth=1.0) == pytest.approx(
        mmd2_plugin(Y, X, bandwidth=1.0)
    )


@pytest.mark.parametrize("n_x,n_y", [(1, 5), (5, 1), (1, 1)])
def test_mmd2_plugin_succeeds_where_mmd2_unbiased_rejects(n_x, n_y):
    rng = np.random.default_rng(91)
    X = rng.normal(size=n_x)
    Y = rng.normal(size=n_y)
    # mmd2_plugin is well-defined at n=1 (true squared norm); mmd2_unbiased
    # is not (its U-statistic denominator is n(n-1)).
    assert np.isfinite(mmd2_plugin(X, Y, bandwidth=1.0))
    with pytest.raises(ValueError):
        mmd2_unbiased(X, Y, bandwidth=1.0)


@pytest.mark.parametrize("n_x,n_y", [(0, 5), (5, 0)])
def test_mmd2_plugin_rejects_empty_sample(n_x, n_y):
    X = np.zeros(n_x)
    Y = np.zeros(n_y)
    with pytest.raises(ValueError):
        mmd2_plugin(X, Y, bandwidth=1.0)


def test_mmd2_plugin_rejects_dimension_mismatch():
    X = np.zeros((4, 2))
    Y = np.zeros((4, 3))
    with pytest.raises(ValueError):
        mmd2_plugin(X, Y, bandwidth=1.0)


@pytest.mark.parametrize("bandwidth", [0.0, -1.0])
def test_mmd2_plugin_rejects_nonpositive_bandwidth(bandwidth):
    X = np.array([0.0, 1.0])
    Y = np.array([2.0, 3.0])
    with pytest.raises(ValueError):
        mmd2_plugin(X, Y, bandwidth=bandwidth)


def test_mmd2_plugin_matches_closed_form_gaussian():
    # Monte Carlo check that the V-statistic plugin and the U-statistic
    # unbiased estimator converge to the same population MMD^2.
    mu1, s1 = 0.0, 1.0
    mu2, s2 = 1.0, 1.5
    sigma = 1.2
    n = 3000
    num_seeds = 5

    plugin_estimates = []
    unbiased_estimates = []
    for seed in range(num_seeds):
        rng = np.random.default_rng(3000 + seed)
        X = rng.normal(mu1, s1, size=n)
        Y = rng.normal(mu2, s2, size=n)
        plugin_estimates.append(mmd2_plugin(X, Y, sigma))
        unbiased_estimates.append(mmd2_unbiased(X, Y, sigma))

    expected = _closed_form_gaussian_mmd2(mu1, s1, mu2, s2, sigma)
    # Both estimators should be close to the same population value; the
    # plugin estimator has an O(1/n) upward bias that vanishes at this n.
    assert np.mean(plugin_estimates) == pytest.approx(expected, abs=0.01)
    assert np.mean(unbiased_estimates) == pytest.approx(expected, abs=0.01)
