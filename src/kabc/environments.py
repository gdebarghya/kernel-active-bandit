"""Bandit environments for Exp 1/2/3 (docs/experiments.md), per CLAUDE.md's
algorithm interface: every environment implements `pull(arm_index, rng) -> sample`
and exposes `n_arms` / `true_partition` so experiment scripts can score a
recovered partition against ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kabc.kernels import median_heuristic_bandwidth, rbf_kernel

_RATING_SUPPORT = np.arange(1, 6, dtype=float).reshape(-1, 1)  # ratings take values {1,...,5}


def _balanced_partition(N: int, K: int) -> np.ndarray:
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K!r}")
    if N % K != 0:
        raise ValueError(f"N must be divisible by K for a balanced partition, got N={N}, K={K}")
    return np.repeat(np.arange(K), N // K)


class Environment:
    """Shared attributes for all environments: `n_arms`, `true_partition`.

    Subclasses set both in `__init__` and implement `pull`.
    """

    n_arms: int
    true_partition: np.ndarray

    def _validate_arm_index(self, arm_index: int) -> int:
        if not 0 <= arm_index < self.n_arms:
            raise ValueError(f"arm_index must be in [0, {self.n_arms}), got {arm_index!r}")
        return int(self.true_partition[arm_index])

    def pull(self, arm_index: int, rng: np.random.Generator):
        raise NotImplementedError


# --------------------------------------------------------------------------
# Experiment 1: Gaussian mixtures
# --------------------------------------------------------------------------

class GaussianMixtureEnv(Environment):
    """Exp 1: N arms in K balanced clusters, arm i in cluster k ~ N(mu_k, sigma^2 I_d).

    Cluster means follow docs/experiments.md's literal formula,
    mu_k = Delta/sqrt(2) * (e_{2k-1} + e_{2k}) (1-indexed k; cluster k occupies
    coordinates 2k-2, 2k-1 zero-indexed). This requires d >= 2*K, since each
    cluster needs its own pair of coordinates -- the doc's own "Sweep d in
    {2, 10, 50, 100}" grid point d=2 is inapplicable whenever K > 1 under this
    formula, and is rejected with a clear error rather than silently
    reinterpreted. Note also that since mu_k and mu_l (k != l) are supported
    on disjoint coordinates (hence orthogonal) with ||mu_k|| = Delta, the
    actual pairwise Euclidean separation this formula produces is
    Delta*sqrt(2), not Delta as docs/experiments.md's prose claims -- kept
    verbatim since the doc is the frozen source of truth for instance
    construction.
    """

    def __init__(self, N: int, K: int, d: int, Delta: float, sigma: float = 1.0):
        if d < 2 * K:
            raise ValueError(
                f"mu_k = Delta/sqrt(2)*(e_{{2k-1}}+e_{{2k}}) requires d >= 2*K "
                f"(each cluster needs its own pair of coordinates); got d={d}, K={K}"
            )
        if Delta <= 0:
            raise ValueError(f"Delta must be > 0, got {Delta!r}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma!r}")

        self.N = N
        self.K = K
        self.d = d
        self.Delta = Delta
        self.sigma = sigma
        self.true_partition = _balanced_partition(N, K)
        self.n_arms = N

        cluster_means = np.zeros((K, d))
        for k in range(K):
            cluster_means[k, 2 * k] = Delta / np.sqrt(2.0)
            cluster_means[k, 2 * k + 1] = Delta / np.sqrt(2.0)
        self.cluster_means = cluster_means

    def pull(self, arm_index: int, rng: np.random.Generator) -> np.ndarray:
        cluster = self._validate_arm_index(arm_index)
        return rng.normal(self.cluster_means[cluster], self.sigma)


# --------------------------------------------------------------------------
# Experiment 2: identical means, different shapes
# --------------------------------------------------------------------------

class ScalePairEnv(Environment):
    """Exp 2a: same-mean, different-variance pair. Cluster A = N(0,1), cluster B = N(0, sigma_B^2)."""

    def __init__(self, N: int, sigma_B: float):
        if sigma_B <= 0:
            raise ValueError(f"sigma_B must be > 0, got {sigma_B!r}")
        self.N = N
        self.sigma_B = sigma_B
        self.true_partition = _balanced_partition(N, 2)
        self.n_arms = N

    def pull(self, arm_index: int, rng: np.random.Generator) -> float:
        cluster = self._validate_arm_index(arm_index)
        sigma = 1.0 if cluster == 0 else self.sigma_B
        return float(rng.normal(0.0, sigma))


class MomentMatchedMixtureEnv(Environment):
    """Exp 2b: cluster A = N(0,1); cluster B = 0.5*N(-a,b^2) + 0.5*N(a,b^2), b^2 = 1-a^2.

    A and B share mean 0 and variance 1 (matched); only the shape (unimodal
    vs bimodal) differs.
    """

    def __init__(self, N: int, a: float):
        if not 0.0 < a < 1.0:
            raise ValueError(f"a must be in (0, 1) so that b^2 = 1 - a^2 > 0, got {a!r}")
        self.N = N
        self.a = a
        self.b = float(np.sqrt(1.0 - a**2))
        self.true_partition = _balanced_partition(N, 2)
        self.n_arms = N

    def pull(self, arm_index: int, rng: np.random.Generator) -> float:
        cluster = self._validate_arm_index(arm_index)
        if cluster == 0:
            return float(rng.normal(0.0, 1.0))
        mean = self.a if rng.integers(2) == 0 else -self.a
        return float(rng.normal(mean, self.b))


class RingBlobEnv(Environment):
    """Exp 2c: ring (radius 2, Gaussian noise 0.1) vs N(0, I_2) blob, both centered at origin.

    Single fixed configuration (d=2), N=40, K=2 per docs/experiments.md.
    Cluster 0 = ring, cluster 1 = blob.
    """

    def __init__(self, N: int = 40, ring_radius: float = 2.0, ring_noise: float = 0.1):
        if ring_radius <= 0:
            raise ValueError(f"ring_radius must be > 0, got {ring_radius!r}")
        if ring_noise <= 0:
            raise ValueError(f"ring_noise must be > 0, got {ring_noise!r}")
        self.N = N
        self.ring_radius = ring_radius
        self.ring_noise = ring_noise
        self.true_partition = _balanced_partition(N, 2)
        self.n_arms = N

    def pull(self, arm_index: int, rng: np.random.Generator) -> np.ndarray:
        cluster = self._validate_arm_index(arm_index)
        if cluster == 0:
            theta = rng.uniform(0.0, 2.0 * np.pi)
            point = self.ring_radius * np.array([np.cos(theta), np.sin(theta)])
            return point + rng.normal(0.0, self.ring_noise, size=2)
        return rng.normal(0.0, 1.0, size=2)


# --------------------------------------------------------------------------
# Experiment 3: MovieLens-1M
# --------------------------------------------------------------------------

def _load_ratings(data_dir: Path) -> pd.DataFrame:
    ratings_path = data_dir / "ratings.dat"
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"{ratings_path} not found. Run `make data` to fetch/verify MovieLens-1M "
            f"before constructing MovieLensEnv."
        )
    return pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        usecols=["movie_id", "rating"],
        encoding="latin-1",
    )


def _load_movie_titles(data_dir: Path) -> dict[int, str]:
    movies_path = data_dir / "movies.dat"
    if not movies_path.exists():
        raise FileNotFoundError(
            f"{movies_path} not found. Run `make data` to fetch/verify MovieLens-1M "
            f"before constructing MovieLensEnv."
        )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        usecols=["movie_id", "title"],
        encoding="latin-1",
    )
    return dict(zip(movies["movie_id"], movies["title"]))


def _movie_rating_distributions(
    ratings: pd.DataFrame, min_ratings: int
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Per-movie empirical rating distribution p_m on {1,...,5} and mean(p_m).

    Only movies with >= min_ratings total ratings are retained (Exp 3 step 1).
    """
    counts = ratings.groupby("movie_id")["rating"].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
    retained = counts[counts.sum(axis=1) >= min_ratings]
    if len(retained) < 2:
        raise ValueError(
            f"only {len(retained)} movie(s) have >= {min_ratings} ratings; "
            f"need at least 2 to run prototype selection"
        )
    probs = retained.div(retained.sum(axis=1), axis=0)
    means = probs @ np.arange(1, 6, dtype=float)
    movie_probs = {int(mid): row.to_numpy(dtype=float) for mid, row in probs.iterrows()}
    movie_means = {int(mid): float(m) for mid, m in means.items()}
    return movie_probs, movie_means


def _prototype_bandwidth(
    ratings: pd.DataFrame, retained_movie_ids, sample_size: int, seed: int
) -> float:
    """Median-heuristic bandwidth on ratings (Exp 3 step 1), from a fixed-seed
    subsample of the retained movies' pooled ratings (the full pool is ~1M
    rows -- infeasible for an O(n^2) pairwise-distance median). Ratings only
    take 5 distinct values, so the median stabilizes at far smaller n."""
    pooled = ratings.loc[ratings["movie_id"].isin(retained_movie_ids), "rating"].to_numpy(dtype=float)
    if pooled.size > sample_size:
        pooled = np.random.default_rng(seed).choice(pooled, size=sample_size, replace=False)
    return median_heuristic_bandwidth(pooled)


def _select_prototypes(
    movie_probs: dict[int, np.ndarray],
    movie_means: dict[int, float],
    K: int,
    mean_gap_tol: float,
    bandwidth: float,
) -> list[int]:
    """Exp 3 steps 2-3: pick K movie prototypes.

    MMD^2 between two movies is computed from their 5-bin rating distributions
    via the RBF Gram matrix on the discrete support {1,...,5} -- algebraically
    identical to kernels.mmd2_plugin on the raw rating samples (grouping the
    double sum over samples by rating value), but O(1) per pair instead of
    O(n^2), needed since all pairs among hundreds of candidate movies are
    considered.
    """
    if K < 2:
        raise ValueError(f"K must be >= 2 (need at least the polarizing-vs-consensus pair), got {K!r}")

    movie_ids = list(movie_probs.keys())
    P = np.stack([movie_probs[mid] for mid in movie_ids])  # (M, 5)
    means = np.array([movie_means[mid] for mid in movie_ids])  # (M,)

    K_vv = rbf_kernel(_RATING_SUPPORT, bandwidth=bandwidth)  # (5, 5)
    self_term = np.einsum("ma,ab,mb->m", P, K_vv, P)
    cross_term = P @ K_vv @ P.T
    mmd2 = self_term[:, None] + self_term[None, :] - 2.0 * cross_term
    np.fill_diagonal(mmd2, -np.inf)

    mean_gap = np.abs(means[:, None] - means[None, :])
    eligible_mmd2 = np.where(mean_gap <= mean_gap_tol, mmd2, -np.inf)
    if not np.any(np.isfinite(eligible_mmd2)):
        raise ValueError(
            f"no movie pair has |mean gap| <= {mean_gap_tol}; cannot select the "
            f"polarizing-vs-consensus pair (Exp 3 step 2)"
        )
    m1, m2 = np.unravel_index(np.argmax(eligible_mmd2), eligible_mmd2.shape)

    selected = [int(m1), int(m2)]
    remaining = [i for i in range(len(movie_ids)) if i not in selected]
    for _ in range(K - 2):
        if not remaining:
            raise ValueError(
                f"only {len(selected)} eligible movie(s) available; cannot select K={K} prototypes"
            )
        min_dist_to_selected = mmd2[np.ix_(remaining, selected)].min(axis=1)
        best = remaining[int(np.argmax(min_dist_to_selected))]
        selected.append(best)
        remaining.remove(best)

    return [movie_ids[i] for i in selected]


class MovieLensEnv(Environment):
    """Exp 3: K movie prototypes from MovieLens-1M; arms resample with replacement
    from their prototype movie's rating pool. Ground truth = prototype identity.

    Prototype selection (deterministic given the data; only the bandwidth
    subsample uses a fixed internal seed) follows docs/experiments.md's
    Exp 3 "Prototype selection" steps 1-4. The default K, arms_per_prototype,
    min_ratings and mean_gap_tol match the doc's stated values; a future
    experiment runner reading configs/*.yaml should override them explicitly
    rather than this module hardcoding config-driven values.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/ml-1m",
        K: int = 4,
        arms_per_prototype: int = 10,
        min_ratings: int = 1000,
        mean_gap_tol: float = 0.05,
        bandwidth_sample_size: int = 5000,
        bandwidth_seed: int = 0,
    ):
        if arms_per_prototype < 1:
            raise ValueError(f"arms_per_prototype must be >= 1, got {arms_per_prototype!r}")

        data_dir = Path(data_dir)
        ratings = _load_ratings(data_dir)
        movie_probs, movie_means = _movie_rating_distributions(ratings, min_ratings)
        bandwidth = _prototype_bandwidth(
            ratings, movie_probs.keys(), bandwidth_sample_size, bandwidth_seed
        )
        prototype_movie_ids = _select_prototypes(movie_probs, movie_means, K, mean_gap_tol, bandwidth)
        titles = _load_movie_titles(data_dir)

        self.K = K
        self.arms_per_prototype = arms_per_prototype
        self.n_arms = K * arms_per_prototype
        self.true_partition = _balanced_partition(self.n_arms, K)
        self.prototype_movie_ids = prototype_movie_ids
        self.prototype_titles = [titles[mid] for mid in prototype_movie_ids]
        self.prototype_bandwidth = bandwidth
        self.prototype_pools = [
            ratings.loc[ratings["movie_id"] == mid, "rating"].to_numpy(dtype=float)
            for mid in prototype_movie_ids
        ]

    def pull(self, arm_index: int, rng: np.random.Generator) -> float:
        cluster = self._validate_arm_index(arm_index)
        return float(rng.choice(self.prototype_pools[cluster]))
