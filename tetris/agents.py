"""Placement-selection agents (PLAN.md §4).

Each agent maps an engine's current state to a legal (rotation, column). All are
deterministic given their seed / weights so trajectories are reproducible.
"""

from __future__ import annotations

import numpy as np

from .engine import CLEAR_POINTS
from .features import HEIGHT, WIDTH, board_features_batch
from .rng import Mulberry32

# Dellacherie's published hand weights over the 8 BCTS features (features 7-8
# zeroed), shipped as a built-in agent (PLAN.md §2).
DELLACHERIE_WEIGHTS = [-1.0, 1.0, -1.0, -1.0, -4.0, -1.0, 0.0, 0.0]

# clear_points as an array for vectorized indexing by lines-cleared.
_CLEAR_POINTS = np.asarray(CLEAR_POINTS, dtype=np.float32)
# Popcount over all 10-bit row values (filled cells per row int).
_POPCOUNT10 = np.array([bin(v).count("1") for v in range(1 << WIDTH)], dtype=np.int64)


def _filled_cells(rows: np.ndarray) -> np.ndarray:
    """(N, 20) uint16 -> (N,) total filled cells per board."""
    return _POPCOUNT10[rows].sum(axis=1)


def _max_height(rows: np.ndarray) -> np.ndarray:
    """(N, 20) uint16 -> (N,) stack max height = 20 - topmost filled row.

    The tallest column's height equals HEIGHT minus the index of the first
    (topmost) row that has any filled cell; 0 for an empty board.
    """
    nz = rows != 0
    any_filled = nz.any(axis=1)
    first_filled = nz.argmax(axis=1)  # 0 when the row is all-empty
    return np.where(any_filled, HEIGHT - first_filled, 0)


def decision_rewards(engine, beta: float):
    """Shared §8 decision-rule inputs for the current piece.

    Returns ``(placements, afters, r)`` where ``afters`` is an ``(P, 20)`` uint16
    array of post-clear afterstate boards and ``r`` the ``(P,)`` float32
    pre-clear decision reward

        r_i = clear_points[lines_i]
              - beta * (0.3 * max(0, dholes_i) + 0.05 * max(0, dmax_height_i))

    matching the training reward (PLAN.md §2/§8) with the terminal -10 excluded
    (the trainer adds it only on the actually-terminal placement). Returns
    ``None`` when there are no legal placements. Used by both the vectorized TD
    self-play loop and :class:`ValueNetAgent`, so the two share one reward.
    """
    placements, feats, afters = engine.candidate_features()
    if not placements:
        return None
    afters_arr = np.asarray(afters, dtype=np.uint16)
    cur = np.asarray(engine.rows, dtype=np.uint16)[None, :]

    # lines_i is recoverable exactly: a placement adds 4 cells; each cleared line
    # removes 10, so lines = (cells_before + 4 - cells_after) / 10.
    cur_cells = int(_filled_cells(cur)[0])
    lines = (cur_cells + 4 - _filled_cells(afters_arr)) // 10
    lines = np.clip(lines, 0, 4).astype(np.int64)
    base = _CLEAR_POINTS[lines]

    holes_after = feats[:, 4]  # feats col 4 = board6[2] = holes
    holes_cur = float(board_features_batch(cur)[0, 2])
    d_holes = holes_after - holes_cur
    d_height = _max_height(afters_arr) - int(_max_height(cur)[0])

    shaping = 0.3 * np.clip(d_holes, 0, None) + 0.05 * np.clip(d_height, 0, None)
    r = base - np.float32(beta) * shaping.astype(np.float32)
    return placements, afters_arr, r.astype(np.float32)


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


class ValueNetAgent:
    """Greedy (optionally epsilon-greedy) afterstate ValueNet agent (PLAN.md §8).

    Scores each legal placement by ``q_i = r_i + gamma * V(A_i)`` in a single
    forward pass over all candidate afterstates and acts by argmax (ties -> first
    in enumeration order, matching :class:`LinearAgent`). At eval time
    ``beta = 0`` (pure clear_points reward) and ``epsilon = 0``; training uses the
    annealed ``beta`` and an ``epsilon`` schedule. torch is imported lazily so
    importing :mod:`tetris.agents` (e.g. in CEM's spawn workers) stays torch-free.
    """

    def __init__(
        self,
        model,
        gamma: float = 0.95,
        beta: float = 0.0,
        epsilon: float = 0.0,
        rng: Mulberry32 | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.rng = rng
        self.device = device

    def act(self, engine) -> tuple[int, int] | None:
        import torch

        from .model import boards_to_tensor

        res = decision_rewards(engine, self.beta)
        if res is None:
            return None
        placements, afters, r = res

        self.model.eval()
        with torch.no_grad():
            values = self.model(boards_to_tensor(afters, self.device))
            values = values.detach().cpu().numpy().reshape(-1)
        q = r + self.gamma * values

        if self.epsilon > 0.0 and self.rng is not None:
            if self.rng.next_float() < self.epsilon:
                idx = int(self.rng.next_float() * len(placements))
                if idx >= len(placements):
                    idx = len(placements) - 1
                return placements[idx]
        return placements[int(np.argmax(q))]
