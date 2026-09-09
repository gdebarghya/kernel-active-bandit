"""Shared types for algorithm modules.

Every algorithm module (kabc.py, acb.py, ...) exposes `fit(env, delta,
budget_cap, rng) -> Result`, per CLAUDE.md's algorithm interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    partition: np.ndarray
    tau: int
    stopped: bool
    history: list[dict] = field(default_factory=list)
