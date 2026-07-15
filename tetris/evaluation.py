"""Shared greedy-evaluation helpers (PLAN.md §4, §6).

A single place for "play N games with a greedy agent on fixed seeds and report
median / mean / p10 lines + pieces", reused by `scripts/evaluate.py`, the
trainers, and `tetris/runio.py`. Games are independent: game i is played on a
caller-supplied seed with a freshly built agent, so results never depend on
game order. Replays record only the move-list — the deterministic engine
reconstructs the rest (see `replay_moves`).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

from .engine import TetrisEngine

# A factory that builds a (greedy) agent for a given game seed. Using a factory
# rather than a single instance keeps stateful agents (e.g. RandomAgent) fully
# reproducible per game and independent of evaluation order.
AgentFactory = Callable[[int], object]


def p10(values: Sequence[float]) -> float:
    """Linear-interpolated 10th percentile (matches scripts/evaluate.py)."""
    s = sorted(values)
    if not s:
        return 0.0
    if len(s) == 1:
        return float(s[0])
    rank = 0.10 * (len(s) - 1)
    lo = int(rank)
    frac = rank - lo
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + frac * (s[hi] - s[lo])


def play_game(
    agent, seed: int, max_pieces: int, record: bool = False
) -> tuple[int, int, list[list[int]] | None]:
    """Play one game to game-over or the piece cap.

    Returns (lines, pieces, moves) where `moves` is the list of chosen
    ``[rotation, column]`` placements when ``record`` is set, else None.
    """
    engine = TetrisEngine(seed=seed)
    moves: list[list[int]] | None = [] if record else None
    while not engine.game_over and engine.pieces < max_pieces:
        move = agent.act(engine)
        if move is None:
            break
        if moves is not None:
            moves.append([int(move[0]), int(move[1])])
        engine.step(*move)
    return engine.lines, engine.pieces, moves


def replay_moves(seed: int, moves: Sequence[Sequence[int]]) -> tuple[int, int]:
    """Reconstruct a game from a seed + recorded move-list.

    Returns (lines, pieces). This is the round-trip inverse of a recorded
    `play_game`: replaying the moves through a fresh engine must reproduce the
    recorded final line/piece counts.
    """
    engine = TetrisEngine(seed=seed)
    for r, c in moves:
        if engine.game_over:
            break
        engine.step(int(r), int(c))
    return engine.lines, engine.pieces


@dataclass
class EvalResult:
    """Aggregate outcome of a fixed-seed evaluation."""

    seeds: list[int]
    lines: list[int]
    pieces: list[int]
    moves: list[list[list[int]] | None]

    @property
    def median_lines(self) -> float:
        return float(statistics.median(self.lines))

    @property
    def mean_lines(self) -> float:
        return float(statistics.mean(self.lines))

    @property
    def p10_lines(self) -> float:
        return p10(self.lines)

    @property
    def mean_pieces(self) -> float:
        return float(statistics.mean(self.pieces))

    @property
    def total_pieces(self) -> int:
        return int(sum(self.pieces))

    @property
    def best_index(self) -> int:
        """Index of the highest-scoring game (ties → first)."""
        return max(range(len(self.lines)), key=lambda i: self.lines[i])


def evaluate(
    make_agent: AgentFactory,
    seeds: Sequence[int],
    max_pieces: int,
    record: bool = False,
) -> EvalResult:
    """Evaluate `make_agent(seed)` across `seeds`, one game per seed."""
    lines: list[int] = []
    pieces: list[int] = []
    moves: list[list[list[int]] | None] = []
    for seed in seeds:
        agent = make_agent(seed)
        ln, pc, mv = play_game(agent, seed, max_pieces, record=record)
        lines.append(int(ln))
        pieces.append(int(pc))
        moves.append(mv)
    return EvalResult(list(seeds), lines, pieces, moves)
