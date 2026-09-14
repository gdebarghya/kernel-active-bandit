"""unif-kkm: uniform sampling + kernel k-means (docs/experiments.md's
"folklore baseline"). Non-adaptive, fixed-budget -- no paper to verify
against, implemented directly per the design doc.

`fit(env, delta, budget_cap, rng) -> Result` matches the standard interface
for uniformity with the other algorithm modules, but `delta` is UNUSED: this
procedure has no confidence-based stopping rule and no delta-PAC guarantee
(contrast kabc/acb/tas_fw, which are all delta-correct) -- it always spends
its entire budget in one shot and always "completes", so there is nowhere
for a confidence level to enter the procedure. `delta` is validated for
type/range like the other algorithms (so a caller's typo doesn't pass
silently) and then explicitly discarded, rather than silently ignored.

Procedure: pull n = budget_cap // N samples per arm (uniform, fixed
schedule, no adaptivity) -- this n is what tau reports, in full, regardless
of anything below. Represent each arm i by its empirical kernel mean
embedding mu_hat_i (RBF kernel, median-heuristic bandwidth from a pooled
sample across all arms, matching kernels.py's convention). Cluster the N
ARMS (not the raw N*n samples) via kernel k-means on the N x N Gram matrix
of arm-embedding inner products, K_arms[i,j] = <mu_hat_i, mu_hat_j>,
computed as the mean of the (n x n) cross-block kernels.rbf_kernel(X_i, X_j,
bandwidth) -- never materializing the full (N*n) x (N*n) Gram matrix, which
would be infeasible for realistic n (e.g. N=120 arms, n=500: a 60000x60000
matrix). K_arms is a genuine Gram matrix of real RKHS points (the arms'
pooled empirical embeddings), hence PSD by construction, exactly the object
kernel k-means's distance formula assumes.

PERFORMANCE NOTE: even avoiding the full (N*n)x(N*n) matrix, each of the
N*(N+1)/2 arm-pair blocks itself costs O(n^2) -- and n = budget_cap // N is
often in the thousands for a realistic s_star_formula-derived budget_cap
(found empirically: N=12, n~2734 didn't finish an isolated run in 120s).
So _arm_kernel_matrix computes each K_arms[i,j] from a bounded random
SUBSAMPLE of each arm's n pulled points (size <= _KERNEL_MATRIX_SAMPLE_SIZE),
not the full n -- this only affects the precision of the K_arms[i,j] Monte
Carlo estimate (still an unbiased estimate of <mu_hat_i, mu_hat_j>, just
noisier for very large n), never tau or which/how many samples were
actually pulled.

Kernel k-means implementation choice: sklearn has no built-in "kernel
k-means" estimator -- only KMeans (needs Euclidean features, not a
precomputed kernel) and SpectralClustering (a DIFFERENT algorithm: Laplacian
eigendecomposition then k-means in the spectral embedding, not the same
iterative Lloyd's-in-feature-space method this baseline is named for).
Rather than add a new dependency (e.g. scikit-learn-extra) for one baseline,
kernel k-means is implemented directly here via the standard closed-form
kernelized distance:
    ||phi(x_i) - mean_c||^2 = K[i,i] - 2/|c| sum_{j in c} K[i,j]
                              + 1/|c|^2 sum_{j,l in c} K[j,l],
with kmeans++-style seeding and multiple random restarts, keeping the
lowest-inertia run.

tau = N*n always; stopped = True always once n >= 1 (non-adaptive: it always
completes on whatever budget it's given). If budget_cap doesn't even afford
one pull per arm (n=0), it returns the same graceful stopped=False fallback
used by kabc.py/acb.py/tas_fw.py for a too-tight cap, rather than crashing.
"""

from __future__ import annotations

import warnings

import numpy as np

from kabc.algorithms import Result
from kabc.kernels import median_heuristic_bandwidth, rbf_kernel

_BANDWIDTH_SAMPLE_SIZE = 2000  # cap for the pooled median-heuristic (O(n^2) pdist)
_KERNEL_MATRIX_SAMPLE_SIZE = 300  # per-arm subsample cap for each O(n^2) arm-pair block
_N_INIT = 10
_MAX_ITER = 100

# Mirrors kabc.py's STOP_REASON_DEGENERATE_PILOT (same underlying failure
# mode: real discrete data, e.g. a highly-rated MovieLens movie, can draw a
# pooled sample whose pairwise distances are all numerically zero). Named
# "_bandwidth" rather than "_pilot" here since unif-kkm has no separate
# pilot phase -- the samples that fail to yield a bandwidth are the SAME
# samples `tau` already paid for and that clustering would otherwise use.
STOP_REASON_DEGENERATE_BANDWIDTH = "degenerate_bandwidth"


def fit(env, delta: float, budget_cap: int, rng: np.random.Generator) -> Result:
    """unif-kkm: uniform sampling + kernel k-means. `delta` is accepted for
    interface conformance and validated, but unused -- see module docstring.
    """
    if not 0.0 < delta <= 1.0:
        raise ValueError(f"delta must be in (0, 1], got {delta!r}")
    # delta is validated above (so a caller's typo doesn't pass silently) but
    # otherwise unused from here on -- see module docstring for why.
    if budget_cap < 1:
        raise ValueError(f"budget_cap must be >= 1, got {budget_cap!r}")

    N = env.n_arms
    K = env.K
    n = budget_cap // N
    if n < 1:
        warnings.warn(
            f"budget_cap={budget_cap} affords fewer than 1 pull per arm (N={N}); "
            f"returning stopped=False with no pulls made (tau=0)",
            stacklevel=2,
        )
        return Result(partition=np.zeros(N, dtype=int), tau=0, stopped=False, history=[])

    arm_samples = [_pull_batch(env, i, n, rng) for i in range(N)]
    tau = N * n

    try:
        bandwidth = _pooled_bandwidth(arm_samples, rng)
    except ValueError:
        # Even the mean-based fallback inside _pooled_bandwidth is
        # degenerate (every pooled sample identical) -- no bandwidth is
        # recoverable from this draw at all. Graceful, not a crash: matches
        # kabc.py's STOP_REASON_DEGENERATE_PILOT convention. tau still
        # reflects the N*n pulls actually made -- unlike kabc's pilot-phase
        # analogue, those pulls already happened and are the real cost of
        # this attempt, degenerate bandwidth or not.
        warnings.warn(
            f"pooled sample's pairwise distances are all (numerically) zero "
            f"(every pooled sample identical); returning stopped=False with "
            f"tau={tau} (the {tau} pulls were made -- only the bandwidth/"
            f"clustering step failed)",
            stacklevel=2,
        )
        return Result(
            partition=np.zeros(N, dtype=int), tau=tau, stopped=False,
            history=[{"stop_reason": STOP_REASON_DEGENERATE_BANDWIDTH}],
        )

    kernel_matrix_samples = [_subsample(X, _KERNEL_MATRIX_SAMPLE_SIZE, rng) for X in arm_samples]
    arm_kernel = _arm_kernel_matrix(kernel_matrix_samples, bandwidth)
    partition = _kernel_kmeans(arm_kernel, K, rng)

    return Result(partition=partition, tau=tau, stopped=True, history=[])


def _pull_batch(env, arm: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([np.atleast_1d(np.asarray(env.pull(arm, rng), dtype=float)) for _ in range(n)])


def _subsample(X: np.ndarray, max_size: int, rng: np.random.Generator) -> np.ndarray:
    if X.shape[0] <= max_size:
        return X
    idx = rng.choice(X.shape[0], size=max_size, replace=False)
    return X[idx]


def _pooled_bandwidth(arm_samples: list[np.ndarray], rng: np.random.Generator) -> float:
    pooled = np.concatenate(arm_samples, axis=0)
    if pooled.shape[0] > _BANDWIDTH_SAMPLE_SIZE:
        idx = rng.choice(pooled.shape[0], size=_BANDWIDTH_SAMPLE_SIZE, replace=False)
        pooled = pooled[idx]
    # on_degenerate="mean": same real-data failure mode documented in
    # kabc.py's _pilot_or_graceful_stop -- a median pairwise distance of
    # exactly zero can happen even when the population has genuine (if
    # rare) spread. Falls back to the mean distance first; `fit` above
    # catches the ValueError this still raises if even THAT is degenerate.
    return median_heuristic_bandwidth(pooled, on_degenerate="mean", warn_on_fallback=True)


def _arm_kernel_matrix(arm_samples: list[np.ndarray], bandwidth: float) -> np.ndarray:
    """K_arms[i,j] = <mu_hat_i, mu_hat_j> = mean(rbf_kernel(X_i, X_j, bandwidth))."""
    N = len(arm_samples)
    arm_kernel = np.empty((N, N))
    for i in range(N):
        for j in range(i, N):
            value = float(rbf_kernel(arm_samples[i], arm_samples[j], bandwidth).mean())
            arm_kernel[i, j] = value
            arm_kernel[j, i] = value
    return arm_kernel


def _kernel_kmeans(arm_kernel: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    best_labels, best_inertia = None, np.inf
    for _ in range(_N_INIT):
        labels, inertia = _kernel_kmeans_single_run(arm_kernel, n_clusters, rng)
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels


def _kernel_kmeans_single_run(
    arm_kernel: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    n = arm_kernel.shape[0]
    diag = np.diag(arm_kernel)

    # kmeans++-style seeding using the kernel-induced squared distance
    # ||phi(x_i)-phi(x_c)||^2 = K[i,i] + K[c,c] - 2*K[i,c].
    centers = [int(rng.integers(n))]
    while len(centers) < n_clusters:
        dist_sq = np.min([diag + diag[c] - 2.0 * arm_kernel[:, c] for c in centers], axis=0)
        dist_sq = np.clip(dist_sq, 0.0, None)
        total = dist_sq.sum()
        probs = dist_sq / total if total > 0 else np.full(n, 1.0 / n)
        centers.append(int(rng.choice(n, p=probs)))

    labels = np.array(
        [int(np.argmin([diag[i] + diag[c] - 2.0 * arm_kernel[i, c] for c in centers])) for i in range(n)]
    )

    dists = np.zeros((n, n_clusters))
    for _ in range(_MAX_ITER):
        for c in range(n_clusters):
            members = labels == c
            size = int(members.sum())
            if size == 0:
                dists[:, c] = np.inf
                continue
            cross = arm_kernel[:, members].sum(axis=1)
            within = arm_kernel[np.ix_(members, members)].sum()
            dists[:, c] = diag - 2.0 / size * cross + within / size**2

        new_labels = np.argmin(dists, axis=1)

        # Re-seed any empty cluster with a currently poorly-fit point
        # (standard practical fix; picks distinct points via argsort).
        empty = [c for c in range(n_clusters) if not np.any(new_labels == c)]
        if empty:
            farthest_order = np.argsort(-np.min(dists, axis=1))
            for c, idx in zip(empty, farthest_order):
                new_labels[idx] = c

        converged = np.array_equal(new_labels, labels)
        labels = new_labels
        if converged:
            break

    inertia = float(np.sum(dists[np.arange(n), labels]))
    return labels, inertia
