"""Stopping-rule thresholds and the doubling-trick schedule (Thuot et al. 2026).

Pure, stateless functions of already-computed scalars (n, delta, N, variance
estimates) -- computing those scalars from raw samples (rkhs_variance,
mmd2_plugin) is kernels.py's job. Loading calibrated constants from
configs/calibration.yaml is a future caller's job, not this module's.

Intended composition, by the not-yet-built src/kabc/algorithms/kabc.py: at
each iteration k, call iteration_confidence then iteration_budget to get
(delta_k, n_k), pull n_k samples per arm, compute rkhs_variance/mmd2_plugin
per pair, then variance_aware_threshold (kabc) or subgaussian_threshold
(kabc-sg, ablation) to decide same-cluster vs different-cluster edges.
"""

from __future__ import annotations

import math


def _log_term(N: int, delta: float) -> float:
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N!r}")
    if not (0.0 < delta <= 1.0):
        raise ValueError(f"delta must be in (0, 1], got {delta!r}")
    return math.log(8.0 * (N**2 - N) / delta)


# --------------------------------------------------------------------------
# stopping thresholds (Eq 6, Discussion)
# --------------------------------------------------------------------------

def variance_aware_threshold(
    n: float,
    delta_prime: float,
    N: int,
    var_i: float,
    var_j: float,
    g_tilde: float = 1.0,
    const: float = 1.0,
) -> float:
    """Variance-aware, Wolfer-Alquier-style threshold B^{i,j}(n, delta') (Eq. 6).

    var_i, var_j are per-arm empirical RKHS variances (kernels.rkhs_variance).
    g_tilde is the kernel's range (sup - inf); 1.0 for the RBF kernel used
    throughout this repo (sup=1 at zero distance, inf->0 as distance->infinity).

    `const` multiplies the ENTIRE returned bound (both terms together), not
    just one inner coefficient -- this is the calibration protocol's "c_cal"
    (docs/experiments.md section 0.2): the whole formula-as-derived is the
    "shape", const=1.0 is the paper's proven theoretical value, and a future
    calibration script freezes a smaller calibrated value in its place.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    if var_i < 0:
        raise ValueError(f"var_i must be >= 0, got {var_i!r}")
    if var_j < 0:
        raise ValueError(f"var_j must be >= 0, got {var_j!r}")
    if g_tilde <= 0:
        raise ValueError(f"g_tilde must be > 0, got {g_tilde!r}")
    if const <= 0:
        raise ValueError(f"const must be > 0, got {const!r}")
    log_term = _log_term(N, delta_prime)
    return const * (
        (math.sqrt(var_i) + math.sqrt(var_j)) * math.sqrt(2.0 * log_term / n)
        + (32.0 / 3.0) * math.sqrt(g_tilde) * log_term / n
    )


def subgaussian_threshold(
    n: float,
    delta_prime: float,
    N: int,
    g_bar: float = 1.0,
    const: float = 1.0,
) -> float:
    """Kernel-uniform sub-Gaussian, Tolstikhin-style threshold (Discussion section).

    The ablation alternative to variance_aware_threshold: does not adapt to
    the per-arm RKHS variance, only to the kernel's bound g_bar (sup g(x,y);
    1.0 for the RBF kernel). Named `delta_prime` (matching
    variance_aware_threshold) since both are literally substitutable at
    CLUSTER's COMPUTETHRESHOLD(n, delta') call site.

    `const` has the same "multiplies the entire bound" semantics as in
    variance_aware_threshold.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    if g_bar <= 0:
        raise ValueError(f"g_bar must be > 0, got {g_bar!r}")
    if const <= 0:
        raise ValueError(f"const must be > 0, got {const!r}")
    log_term = _log_term(N, delta_prime)
    return const * math.sqrt(g_bar / n) * (math.sqrt(log_term) + 2.0)


# --------------------------------------------------------------------------
# doubling-trick schedule (Algorithm 2)
# --------------------------------------------------------------------------

def iteration_confidence(k: int, delta: float) -> float:
    """Per-iteration confidence delta_k = delta / (4*k**2) (Algorithm 2)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    if not (0.0 < delta <= 1.0):
        raise ValueError(f"delta must be in (0, 1], got {delta!r}")
    return delta / (4.0 * k**2)


def iteration_budget(k: int, N: int, delta_k: float) -> int:
    """Per-arm sampling budget n_k = ceil(2**k * log(8(N^2-N)/delta_k)) (Algorithm 2)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    return math.ceil(2.0**k * _log_term(N, delta_k))


def oracle_budget(s_star_sq: float, N: int, delta: float, const: float = 1.0) -> int:
    """Non-adaptive per-arm budget n* for kabc-oracle (Discussion section).

    s_star_sq is a given signal-to-noise-ratio quantity (an analytic property
    of a known instance) -- computing it is out of scope for this module.

    `const` is included for API symmetry with the two threshold functions
    above (default 1.0, the theoretical constant 128); whether a future
    calibration script should ever override it is a decision deferred to
    whoever wires up Experiment 2's price-of-adaptivity metric
    (tau(kabc)/tau(kabc-oracle)) -- not decided here.
    """
    if s_star_sq <= 0:
        raise ValueError(f"s_star_sq must be > 0, got {s_star_sq!r}")
    if const <= 0:
        raise ValueError(f"const must be > 0, got {const!r}")
    return math.ceil(const * 128.0 / s_star_sq * _log_term(N, delta))
