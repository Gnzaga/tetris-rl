"""Evaluate an agent over N games (PLAN.md §4).

Reports median / mean / p10 of lines and pieces. Game i uses seed = base_seed+i.
Supports a --max-pieces cap (game ends at the cap, reporting lines-at-cap).
"""

import _pathshim  # noqa: F401
import argparse
import json
import statistics

from tetris.agents import DELLACHERIE_WEIGHTS, LinearAgent, RandomAgent
from tetris.evaluation import evaluate, p10


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["random", "dellacherie", "linear"], required=True)
    ap.add_argument("--weights", default=None, help="JSON path for --agent linear")
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7, help="base seed; game i uses seed+i")
    ap.add_argument("--max-pieces", type=int, default=50000)
    args = ap.parse_args()

    seeds = [args.seed + i for i in range(args.games)]
    result = evaluate(
        lambda s: make_agent(args.agent, args.weights, seed=s),
        seeds,
        args.max_pieces,
    )
    for i, seed in enumerate(seeds):
        print(
            f"game {i:>3}  seed={seed:>6}  "
            f"lines={result.lines[i]:>8}  pieces={result.pieces[i]:>8}"
        )

    print("-" * 48)
    print(f"agent            : {args.agent}")
    print(f"games            : {args.games}")
    print(f"lines  median    : {result.median_lines:.1f}")
    print(f"lines  mean      : {result.mean_lines:.1f}")
    print(f"lines  p10       : {result.p10_lines:.1f}")
    print(f"pieces median    : {statistics.median(result.pieces):.1f}")
    print(f"pieces mean      : {result.mean_pieces:.1f}")
    print(f"pieces p10       : {p10(result.pieces):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
