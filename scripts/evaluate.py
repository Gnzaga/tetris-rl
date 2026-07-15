"""Evaluate an agent over N games (PLAN.md §4).

Reports median / mean / p10 of lines and pieces. Game i uses seed = base_seed+i.
Supports a --max-pieces cap (game ends at the cap, reporting lines-at-cap).
"""

import _pathshim  # noqa: F401
import argparse
import json
import statistics

from tetris.agents import DELLACHERIE_WEIGHTS, LinearAgent, RandomAgent
from tetris.engine import TetrisEngine


def play_game(agent, seed: int, max_pieces: int) -> tuple[int, int]:
    """Play one game; return (lines, pieces)."""
    engine = TetrisEngine(seed=seed)
    while not engine.game_over and engine.pieces < max_pieces:
        move = agent.act(engine)
        if move is None:
            break
        engine.step(*move)
    return engine.lines, engine.pieces


def make_agent(kind: str, weights_path: str | None, seed: int):
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "dellacherie":
        return LinearAgent(DELLACHERIE_WEIGHTS)
    if kind == "linear":
        if not weights_path:
            raise SystemExit("--weights is required for --agent linear")
        with open(weights_path) as f:
            data = json.load(f)
        weights = data["weights"] if isinstance(data, dict) else data
        return LinearAgent(weights)
    raise SystemExit(f"unknown agent: {kind}")


def _p10(values: list[int]) -> float:
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    # linear-interpolated 10th percentile
    rank = 0.10 * (len(s) - 1)
    lo = int(rank)
    frac = rank - lo
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + frac * (s[hi] - s[lo])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["random", "dellacherie", "linear"], required=True)
    ap.add_argument("--weights", default=None, help="JSON path for --agent linear")
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7, help="base seed; game i uses seed+i")
    ap.add_argument("--max-pieces", type=int, default=50000)
    args = ap.parse_args()

    lines_list = []
    pieces_list = []
    for i in range(args.games):
        agent = make_agent(args.agent, args.weights, seed=args.seed + i)
        lines, pieces = play_game(agent, seed=args.seed + i, max_pieces=args.max_pieces)
        lines_list.append(lines)
        pieces_list.append(pieces)
        print(f"game {i:>3}  seed={args.seed + i:>6}  lines={lines:>8}  pieces={pieces:>8}")

    print("-" * 48)
    print(f"agent            : {args.agent}")
    print(f"games            : {args.games}")
    print(f"lines  median    : {statistics.median(lines_list):.1f}")
    print(f"lines  mean      : {statistics.mean(lines_list):.1f}")
    print(f"lines  p10       : {_p10(lines_list):.1f}")
    print(f"pieces median    : {statistics.median(pieces_list):.1f}")
    print(f"pieces mean      : {statistics.mean(pieces_list):.1f}")
    print(f"pieces p10       : {_p10(pieces_list):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
