"""Shared metrics: partition error/ARI vs ground truth, and a general-purpose
Monte Carlo estimator of the signal-to-noise ratio s*^2 (Eq. 7,
arXiv:2601.07535) used to size budget_cap for synthetic experiments.

Reused by scripts/common.py's `_resolve_budget_cap` (per-grid-point
budget_cap, per docs/experiments.md section 0.4, shared by the
run_experiment1/2/3.py dedicated scripts) and scripts/calibrate_constants.py
(a generous fixed budget_cap for calibration runs, independent of the c_cal
candidate under test).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from kabc.kernels import median_heuristic_bandwidth, mmd2_plugin, rkhs_variance


def ari(partition: np.ndarray, true_partition: np.ndarray) -> float:
    """Adjusted Rand Index of a recovered partition against ground truth."""
    return float(adjusted_rand_score(true_partition, partition))


def error(partition: np.ndarray, true_partition: np.ndarray) -> int:
    """1{partition != ground truth}, up to permutation of cluster labels
    (docs/experiments.md section 0.4). ARI == 1.0 exactly iff the two
    partitions induce the same equivalence relation on arms (same groupings,
    any relabeling) -- exactly the intended check.
    """
    return 0 if ari(partition, true_partition) == 1.0 else 1


def estimate_s_star_sq(
    env,
    rng: np.random.Generator,
    n_oracle_sample: int = 3000,
    n_pilot: int = 5,
) -> float:
    """Monte Carlo estimate of the instance's TRUE signal-to-noise ratio
    s*^2 (Eq. 7): min over pairs of distinct true clusters i != j of
        ( ||mu_i-mu_j||^2 / (V_i v V_j) ) ^ ( 2||mu_i-mu_j|| / sqrt(g_bar) ),
    where ^ here is "and" (the min of the two terms), matching Eq. 7's
    notation.

    Uses a large, UNBUDGETED auxiliary sample per true cluster (one
    representative arm each, since arms within a cluster share a
    distribution) -- oracle-only information a real algorithm wouldn't have,
    used here purely to size budget_cap for the experiment grid; never
    counted toward any algorithm's own tau. General-purpose (Monte Carlo,
    not closed-form): works for every environment family in this repo, not
    just Gaussians -- matches docs/experiments.md's own "compute s* ...
    numerically" phrasing (Exp 1, Fig. 1 note).

    Bandwidth is estimated the same way an algorithm's own pilot phase would
    (n_pilot pulls/arm, pooled, median heuristic), for consistency with what
    the actual experiment run will use.
    """
    N = env.n_arms
    K = env.K
    g_bar = 1.0  # RBF kernel's sup

    pilot = [
        np.stack([np.atleast_1d(np.asarray(env.pull(i, rng), dtype=float)) for _ in range(n_pilot)])
        for i in range(N)
    ]
    bandwidth = median_heuristic_bandwidth(np.concatenate(pilot, axis=0))

    cluster_samples = []
    for k in range(K):
        arm = int(np.where(env.true_partition == k)[0][0])
        cluster_samples.append(
            np.stack(
                [np.atleast_1d(np.asarray(env.pull(arm, rng), dtype=float)) for _ in range(n_oracle_sample)]
            )
        )

    s_star_sq = np.inf
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            mmd2 = mmd2_plugin(cluster_samples[i], cluster_samples[j], bandwidth)
            v_i = rkhs_variance(cluster_samples[i], bandwidth)
            v_j = rkhs_variance(cluster_samples[j], bandwidth)
            gap_term = mmd2 / max(v_i, v_j)
            kernel_term = 2.0 * np.sqrt(mmd2) / np.sqrt(g_bar)
            s_star_sq = min(s_star_sq, gap_term, kernel_term)
    return float(s_star_sq)
