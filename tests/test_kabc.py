import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from kabc.algorithms.kabc import (
    MEMORY_GUARD_MATRICES_PER_CALL,
    MEMORY_GUARD_N_WORKERS,
    MEMORY_GUARD_SAFE_TOTAL_BYTES,
    MEMORY_GUARD_SAFETY_FACTOR,
    N_PILOT,
    STOP_REASON_BUDGET_CAP,
    STOP_REASON_DEGENERATE_PILOT,
    STOP_REASON_MAX_ITERATIONS,
    STOP_REASON_MEMORY_GUARD,
    _check_memory_guard,
    _fit,
    _MemoryGuardExceeded,
    fit,
    fit_sg,
)
from kabc.environments import GaussianMixtureEnv
from kabc.thresholds import iteration_budget, iteration_confidence


class _MostlyTiedEnv:
    """Test double: N-1 arms always pull an identical value, one arm pulls a
    different (but still identical-across-its-own-pulls) value. Engineered
    so the pooled pilot sample's MEDIAN pairwise distance is exactly zero
    (a majority of pairs are tied-value pairs) while the MEAN is not (a
    minority of pairs cross the two values) -- reproducing the real
    Experiment 4 incident (Schindler's List vs Rear Window: a small pilot
    sample from two highly-rated, low-variance movies drew mostly 4s/5s by
    chance) at a scale small enough for a deterministic unit test.
    """

    def __init__(self, N: int, K: int = 3):
        self.n_arms = N
        self.K = K
        self.true_partition = np.zeros(N, dtype=int)

    def pull(self, arm_index: int, rng: np.random.Generator) -> float:
        return 3.0 if arm_index == 0 else 5.0


class _FullyDegenerateEnv:
    """Test double: EVERY arm always pulls the identical value -- both the
    median AND mean pairwise pilot distance are zero, so no bandwidth is
    recoverable from this data at all (unlike _MostlyTiedEnv above, where
    the mean fallback succeeds)."""

    def __init__(self, N: int, K: int = 3):
        self.n_arms = N
        self.K = K
        self.true_partition = np.zeros(N, dtype=int)

    def pull(self, arm_index: int, rng: np.random.Generator) -> float:
        return 5.0


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
    assert result.history == [{"stop_reason": STOP_REASON_BUDGET_CAP}]
    assert result.tau == pilot_cost
    assert result.partition.shape == (N,)
    # best-effort fallback: no iteration ran, so all arms are reported as one cluster
    assert np.all(result.partition == result.partition[0])


def test_degenerate_pilot_bandwidth_falls_back_to_mean_and_still_converges():
    # Real incident (Experiment 4): a small pilot sample from two low-variance
    # movies drew mostly one rating value by chance, making the MEDIAN
    # pairwise pilot distance zero even though the data has real (if rare)
    # spread. This must fall back to the mean pairwise distance and proceed
    # normally, not raise -- see kernels.median_heuristic_bandwidth's
    # on_degenerate="mean" and kabc.py's _pilot_or_graceful_stop.
    env = _MostlyTiedEnv(N=12, K=3)
    rng = np.random.default_rng(0)
    with pytest.warns(UserWarning, match="fell back to the mean pairwise distance"):
        result = fit(env, delta=0.05, budget_cap=50_000, rng=rng)
    # Recovers a real bandwidth and keeps going -- not the graceful
    # degenerate-pilot stop (that path is for when even the mean fails).
    assert result.history[-1].get("stop_reason") != STOP_REASON_DEGENERATE_PILOT


def test_stop_reason_is_degenerate_pilot_when_mean_fallback_also_fails():
    # Every pooled pilot sample is identical -- no bandwidth is recoverable
    # from this data at all, even with the mean-distance backup. Must
    # degrade gracefully (stopped=False, a distinctly labeled stop_reason),
    # never raise and crash the caller's run.
    env = _FullyDegenerateEnv(N=12, K=3)
    N = env.n_arms
    pilot_cost = N * N_PILOT
    rng = np.random.default_rng(0)

    with pytest.warns(UserWarning, match="only the pilot phase completed"):
        result = fit(env, delta=0.05, budget_cap=50_000, rng=rng)

    assert result.stopped is False
    assert result.history == [{"stop_reason": STOP_REASON_DEGENERATE_PILOT}]
    assert result.tau == pilot_cost
    assert result.partition.shape == (N,)
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
    assert len(result.history) == 2
    assert result.history[0]["k"] == 1
    assert result.history[1] == {"stop_reason": STOP_REASON_BUDGET_CAP}
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
    assert result.history == [{"stop_reason": STOP_REASON_BUDGET_CAP}]
    assert result.partition.shape == (N,)
    assert np.all(result.partition == result.partition[0])


# --------------------------------------------------------------------------
# graceful-stop reasons are distinctly labeled (budget_cap / max_iterations
# / memory_guard), not collapsed into one generic "didn't converge"
# --------------------------------------------------------------------------

def test_stop_reason_is_budget_cap_when_iteration_unaffordable():
    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    N = env.n_arms
    pilot_cost = N * N_PILOT
    delta_1 = iteration_confidence(1, 0.05)
    n_1 = iteration_budget(1, N, delta_1)
    tiny_cap = pilot_cost + N * n_1 - 1  # affords the pilot, not iteration k=1

    rng = np.random.default_rng(0)
    result = fit(env, delta=0.05, budget_cap=tiny_cap, rng=rng)

    assert result.stopped is False
    assert result.history[-1] == {"stop_reason": STOP_REASON_BUDGET_CAP}


def test_stop_reason_is_max_iterations_when_schedule_never_converges_or_caps():
    # An instance with NO true cluster structure (K=1 arms all overlapping
    # centers is awkward to construct via GaussianMixtureEnv; instead use a
    # budget_cap generous enough that budget_cap never binds, and a K that
    # can never actually be matched by connected components on random noise
    # at small n -- forcing the loop through MAX_ITERATIONS iterations
    # without ever hitting `num_clusters == K`) is fragile to rely on
    # statistically, so this test instead directly shrinks MAX_ITERATIONS
    # via monkeypatching the module constant, to deterministically exercise
    # the backstop without needing a pathological real instance.
    import kabc.algorithms.kabc as kabc_module

    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    rng = np.random.default_rng(0)

    original_max_iterations = kabc_module.MAX_ITERATIONS
    kabc_module.MAX_ITERATIONS = 0  # every k > 0, so the loop stops at k=1 immediately
    try:
        result = _fit(env, delta=0.05, budget_cap=50_000, rng=rng, threshold="variance-aware")
    finally:
        kabc_module.MAX_ITERATIONS = original_max_iterations

    assert result.stopped is False
    assert result.history == [{"stop_reason": STOP_REASON_MAX_ITERATIONS}]


def test_stop_reason_is_memory_guard_when_gram_matrices_would_be_unsafe():
    # Directly exercises the guard's own logic (see its docstring for the
    # formula) rather than trying to construct a real instance whose
    # doubling-trick schedule happens to reach a huge n_k -- that would be
    # slow and indirect for a unit test whose job is to check the guard
    # trips and is labeled correctly, not to re-derive a pathological
    # calibration scenario.
    max_safe_n = int(
        (MEMORY_GUARD_SAFE_TOTAL_BYTES / (MEMORY_GUARD_N_WORKERS * MEMORY_GUARD_MATRICES_PER_CALL * MEMORY_GUARD_SAFETY_FACTOR * 8))
        ** 0.5
    )
    _check_memory_guard(max_safe_n)  # should not raise: just under the safe budget

    with pytest.raises(_MemoryGuardExceeded):
        _check_memory_guard(max_safe_n * 10)  # far over budget: must raise

    # And confirm _fit surfaces this as a correctly labeled, graceful stop --
    # not a crash -- when the doubling trick's very first iteration already
    # demands an unsafe n_k. GaussianMixtureEnv with an extremely small Delta
    # (near-zero separation) pushes n_1 up via iteration_budget, but the
    # cleanest deterministic way to force it is the same monkeypatch
    # approach as the max_iterations test: shrink MEMORY_GUARD_SAFE_TOTAL_BYTES
    # so the ordinary k=1 n_1 already exceeds it.
    import kabc.algorithms.kabc as kabc_module

    env = GaussianMixtureEnv(N=12, K=3, d=6, Delta=8.0)
    rng = np.random.default_rng(0)

    original_budget = kabc_module.MEMORY_GUARD_SAFE_TOTAL_BYTES
    kabc_module.MEMORY_GUARD_SAFE_TOTAL_BYTES = 1  # 1 byte: any real n_k exceeds this
    try:
        result = _fit(env, delta=0.05, budget_cap=50_000, rng=rng, threshold="variance-aware")
    finally:
        kabc_module.MEMORY_GUARD_SAFE_TOTAL_BYTES = original_budget

    assert result.stopped is False
    assert result.history[0]["stop_reason"] == STOP_REASON_MEMORY_GUARD
    assert result.history[0]["k"] == 1
