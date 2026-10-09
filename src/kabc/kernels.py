"""RBF kernel, median-heuristic bandwidth, and MMD^2 estimation.

Note for whoever wires this into the `kabc` main loop / `thresholds.py`: every
call here recomputes full Gram matrices from scratch, O(n^2) in the current
sample size. As arm sample sizes grow into the thousands over a run, calling
`mmd2_and_variance` on the full pull history at every stopping-rule check will
get expensive — consider an infrequent (e.g. doubling-schedule) recomputation
rather than recomputing on every pull.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel as _sklearn_rbf_kernel


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"expected a 1D or 2D array, got shape {X.shape}")
    return X


def _validate_bandwidth(bandwidth: float) -> None:
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(f"bandwidth must be a positive finite float, got {bandwidth!r}")


def rbf_kernel(X: np.ndarray, Y: np.ndarray | None = None, bandwidth: float = 1.0) -> np.ndarray:
    """Gaussian RBF Gram matrix: k(x,y) = exp(-||x-y||^2 / (2*bandwidth**2)).

    `bandwidth` is sigma (not sklearn's gamma = 1/(2*sigma**2)); accepts 1D
    arrays of shape (n,) as well as 2D arrays of shape (n,d). If Y is None,
    computes the Gram matrix of X against itself.
    """
    _validate_bandwidth(bandwidth)
    X = _as_2d(X)
    Y = X if Y is None else _as_2d(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same feature dimension; got {X.shape[1]} and {Y.shape[1]}"
        )
    gamma = 1.0 / (2.0 * bandwidth**2)
    return _sklearn_rbf_kernel(X, Y, gamma=gamma)


def median_heuristic_bandwidth(X: np.ndarray, on_degenerate: str = "raise", warn_on_fallback: bool = False) -> float:
    """Median-heuristic RBF bandwidth from a pooled (pilot) sample.

    Returns the median of all pairwise Euclidean distances in X (excluding
    self-pairs), used directly as `bandwidth` (sigma) in `rbf_kernel`.

    `on_degenerate` controls behavior when the MEDIAN pairwise distance is
    (numerically) zero -- i.e. more than half of all pairs are exactly tied.
    Real discrete data can trigger this even when the underlying population
    has genuine spread: e.g. a small pilot sample from a highly-rated
    MovieLens movie can draw mostly 4s/5s by chance (the low ratings are
    rare, not absent), giving a pooled sample dominated by one tied value.
    - `"raise"` (default): original behavior, preserved for every existing
      caller unless they opt in otherwise.
    - `"mean"`: fall back to the MEAN pairwise distance instead. The mean
      stays informative even when a MAJORITY of pairs are tied (unlike the
      median, which only needs >50% zero pairs to collapse) -- it only
      raises if the mean is ALSO (numerically) zero, meaning every pooled
      sample is identical and no bandwidth is recoverable from this data at
      all.

    `warn_on_fallback`: if True, emits a UserWarning when the `"mean"`
    fallback actually triggers (never when the median succeeds normally),
    so a caller relying on the backup path stays visible/traceable in logs
    rather than silently changing behavior -- matching this repo's existing
    convention for other graceful-degradation paths (e.g.
    `kabc.py`'s `_pilot_or_graceful_stop` budget-too-tight warning).
    """
    X = _as_2d(X)
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"median_heuristic_bandwidth requires at least 2 pooled samples, got {n}")
    if on_degenerate not in ("raise", "mean"):
        raise ValueError(f"on_degenerate must be 'raise' or 'mean', got {on_degenerate!r}")
    distances = pdist(X, metric="euclidean")
    median = float(np.median(distances))
    if median >= 1e-12:
        return median
    if on_degenerate == "raise":
        raise ValueError(
            "median pairwise distance is (numerically) zero; pooled pilot sample "
            "appears degenerate (duplicate/identical points) — cannot set an RBF "
            "bandwidth from it"
        )
    mean = float(np.mean(distances))
    if mean < 1e-12:
        raise ValueError(
            "median AND mean pairwise distance are both (numerically) zero; "
            "every pooled sample is identical — no bandwidth is recoverable "
            "from this data at all"
        )
    if warn_on_fallback:
        warnings.warn(
            f"median pairwise distance was (numerically) zero (pooled sample "
            f"dominated by tied values); fell back to the mean pairwise "
            f"distance ({mean:.6g}) as the bandwidth instead of raising",
            stacklevel=2,
        )
    return mean


def _gram_matrices(X: np.ndarray, Y: np.ndarray, bandwidth: float):
    X = _as_2d(X)
    Y = _as_2d(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same feature dimension; got {X.shape[1]} and {Y.shape[1]}"
        )
    n_x, n_y = X.shape[0], Y.shape[0]
    if n_x < 2 or n_y < 2:
        raise ValueError(
            f"mmd2 estimators require at least 2 samples in each of X and Y, "
            f"got n_x={n_x}, n_y={n_y}"
        )
    Kxx = rbf_kernel(X, X, bandwidth)
    Kyy = rbf_kernel(Y, Y, bandwidth)
    Kxy = rbf_kernel(X, Y, bandwidth)
    return Kxx, Kyy, Kxy, n_x, n_y


def _mmd2_from_grams(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray, n_x: int, n_y: int) -> float:
    term_x = (Kxx.sum() - np.trace(Kxx)) / (n_x * (n_x - 1))
    term_y = (Kyy.sum() - np.trace(Kyy)) / (n_y * (n_y - 1))
    term_xy = Kxy.sum() / (n_x * n_y)
    return float(term_x + term_y - 2.0 * term_xy)


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> float:
    """Unbiased U-statistic MMD^2 estimator (Gretton et al. 2012), n_x != n_y OK.

    Not clipped to be non-negative: this estimator is legitimately negative for
    finite samples under/near the null (X, Y from the same distribution).
    """
    Kxx, Kyy, Kxy, n_x, n_y = _gram_matrices(X, Y, bandwidth)
    return _mmd2_from_grams(Kxx, Kyy, Kxy, n_x, n_y)


def mmd2_and_variance(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> tuple[float, float]:
    """Unbiased MMD^2 estimate together with a variance estimate of that estimator.

    The variance estimate is a first-order (Hajek-projection) approximation
    from standard two-sample degree-2 U-statistic asymptotics: for the
    leave-one-out within-sample means g_X, g_Y and the sample cross-means g_1,
    g_2, with psi_X = g_X - g_1 and psi_Y = g_Y - g_2,

        Var_hat = (4/n_x) * Var(psi_X) + (4/n_y) * Var(psi_Y).

    Both coefficients are 4 (not 4 and 1): the cross term's factor-1
    contribution is doubled by MMD^2's -2x cross-term coefficient, exactly
    matching the within-sample terms' factor of 2 from their degree-2 U-statistic
    projection. This is an asymptotic approximation, most accurate for
    moderate-to-large n_x, n_y.
    """
    Kxx, Kyy, Kxy, n_x, n_y = _gram_matrices(X, Y, bandwidth)
    mmd2 = _mmd2_from_grams(Kxx, Kyy, Kxy, n_x, n_y)

    g_X = (Kxx.sum(axis=1) - np.diag(Kxx)) / (n_x - 1)
    g_1 = Kxy.sum(axis=1) / n_y
    psi_X = g_X - g_1

    g_Y = (Kyy.sum(axis=1) - np.diag(Kyy)) / (n_y - 1)
    g_2 = Kxy.sum(axis=0) / n_x
    psi_Y = g_Y - g_2

    variance = 4.0 / n_x * np.var(psi_X, ddof=1) + 4.0 / n_y * np.var(psi_Y, ddof=1)
    return mmd2, float(variance)


def rkhs_variance(X: np.ndarray, bandwidth: float) -> float:
    """Empirical RKHS variance of a single arm's distribution (Thuot et al. 2026, Eq. (3)).

    V_hat_i = (trace(K) - K.sum()/n) / (n-1), where K = rbf_kernel(X, X, bandwidth).

    This is a per-arm spread estimate in the RKHS, NOT the variance of a
    two-sample statistic — it is a different quantity from
    `mmd2_and_variance`'s variance-of-the-MMD^2-estimator, computed from a
    single sample rather than a pair. It is the input `thresholds.py`'s
    variance-aware threshold consumes.

    Provably >= 0 for the RBF kernel: g(x,x)=1 is the kernel's global sup, so
    each term g(x_t,x_t) - mean_s(g(x_t,x_s)) is individually >= 0.
    """
    X = _as_2d(X)
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"rkhs_variance requires at least 2 samples, got n={n}")
    _validate_bandwidth(bandwidth)
    K = rbf_kernel(X, X, bandwidth)
    return float((np.trace(K) - K.sum() / n) / (n - 1))


def mmd2_plugin(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> float:
    """Empirical squared MMD as the plain plug-in/V-statistic (Thuot et al. 2026, Eq. (2)).

    ||mu_hat_i - mu_hat_j||^2, computed via the reproducing property from the
    full (diagonal-included) double sum over kernel evaluations. Unlike
    `mmd2_unbiased`'s Gretton U-statistic (which excludes the diagonal and can
    be negative), this is a true squared norm: always >= 0 (clipped at 0.0 to
    guard against floating-point noise only), and well-defined even at
    n_x=1/n_y=1 (unlike the unbiased estimator's n(n-1) denominator).

    Generalizes the paper's equal-sample-size formula to n_x != n_y (each of
    the three terms gets its own n_x**2 / n_y**2 / n_x*n_y normalizer); reduces
    exactly to the paper's Eq. (2) when n_x == n_y.
    """
    X = _as_2d(X)
    Y = _as_2d(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same feature dimension; got {X.shape[1]} and {Y.shape[1]}"
        )
    n_x, n_y = X.shape[0], Y.shape[0]
    if n_x < 1 or n_y < 1:
        raise ValueError(
            f"mmd2_plugin requires at least 1 sample in each of X and Y, "
            f"got n_x={n_x}, n_y={n_y}"
        )
    Kxx = rbf_kernel(X, X, bandwidth)
    Kyy = rbf_kernel(Y, Y, bandwidth)
    Kxy = rbf_kernel(X, Y, bandwidth)
    value = Kxx.sum() / n_x**2 - 2.0 * Kxy.sum() / (n_x * n_y) + Kyy.sum() / n_y**2
    return float(max(value, 0.0))
