import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from kabc.algorithms.tas_fw import fit


class _TinyCategoricalEnv:
    """Self-contained, inline finite-alphabet environment for testing only --
    N=6 arms in K=3 well-separated clusters over alphabet {0,1,2}, matching
    Exp 3's reduced-instance scale (docs/experiments.md), not Exp 1 (tas-fw
    is not run there; see the module docstring in tas_fw.py). No
    environments.py changes or MovieLens dependency needed for this test.
    """

    def __init__(self):
        self.n_arms = 6
        self.K = 3
        self.true_partition = np.repeat(np.arange(3), 2)
        self.alphabet = [0, 1, 2]
        self.cluster_probs = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]

    def pull(self, arm_index: int, rng: np.random.Generator) -> int:
        cluster = self.true_partition[arm_index]
        return int(rng.choice(self.alphabet, p=self.cluster_probs[cluster]))


def test_recovers_partition_on_small_well_separated_instance():
    env = _TinyCategoricalEnv()
    rng = np.random.default_rng(0)
    result = fit(env, delta=0.1, budget_cap=20_000, rng=rng)

    assert result.stopped is True
    assert adjusted_rand_score(env.true_partition, result.partition) == pytest.approx(1.0)


def test_stops_gracefully_at_budget_cap():
    env = _TinyCategoricalEnv()
    rng = np.random.default_rng(0)
    result = fit(env, delta=0.1, budget_cap=50, rng=rng)

    assert result.stopped is False
    assert result.tau == 50
    assert result.partition.shape == (6,)
