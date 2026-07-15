"""Placement-selection agents (PLAN.md §4).

Each agent maps an engine's current state to a legal (rotation, column). All are
deterministic given their seed / weights so trajectories are reproducible.
"""

from __future__ import annotations

import numpy as np

from .rng import Mulberry32

# Dellacherie's published hand weights over the 8 BCTS features (features 7-8
# zeroed), shipped as a built-in agent (PLAN.md §2).
DELLACHERIE_WEIGHTS = [-1.0, 1.0, -1.0, -1.0, -4.0, -1.0, 0.0, 0.0]


class RandomAgent:
    """Picks uniformly among legal placements using its own seeded PRNG."""

    def __init__(self, seed: int = 0, rng: Mulberry32 | None = None):
        self.rng = rng if rng is not None else Mulberry32(seed)

    def act(self, engine) -> tuple[int, int] | None:
        placements = engine.legal_placements()
        if not placements:
            return None
        idx = int(self.rng.next_float() * len(placements))
        if idx >= len(placements):  # guard the float==1.0 edge
            idx = len(placements) - 1
        return placements[idx]


class LinearAgent:
    """Scores each legal placement by w·f and picks the argmax.

    Tie-break: np.argmax returns the first maximal index, i.e. the first
    placement in enumeration order (rotation asc, column asc) — deterministic.
    """

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.shape != (8,):
            raise ValueError(f"expected 8 weights, got {self.weights.shape}")

    def act(self, engine) -> tuple[int, int] | None:
        placements, feats, _ = engine.candidate_features()
        if not placements:
            return None
        scores = feats @ self.weights
        return placements[int(np.argmax(scores))]


def dellacherie_agent() -> LinearAgent:
    """The built-in Dellacherie hand-tuned linear agent."""
    return LinearAgent(DELLACHERIE_WEIGHTS)
