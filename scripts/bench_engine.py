"""Benchmark full decision cycles (PLAN.md §4, gate 2).

A "cycle" enumerates all legal placements for a state and computes all 8 BCTS
features for each. The gate requires >= 1,000 cycles/sec/core. We build a pool
of realistic mid-game boards via random self-play, then time repeated
`candidate_features` calls over that pool.
"""

import _pathshim  # noqa: F401
import argparse
import time

from tetris.agents import RandomAgent
from tetris.engine import TetrisEngine


def build_state_pool(n_states: int, seed: int) -> list[tuple[list[int], int]]:
    """Snapshot (rows, current_piece) from random self-play across games."""
    pool: list[tuple[list[int], int]] = []
    game_seed = seed
    while len(pool) < n_states:
        engine = TetrisEngine(seed=game_seed)
        agent = RandomAgent(seed=game_seed ^ 0x9E3779B9)
        game_seed += 1
        steps = 0
        while not engine.game_over and steps < 400:
            pool.append((list(engine.rows), engine.current))
            move = agent.act(engine)
            if move is None:
                break
            engine.step(*move)
            steps += 1
            if len(pool) >= n_states:
                break
    return pool[:n_states]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=500)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    pool = build_state_pool(args.states, args.seed)
    engine = TetrisEngine(seed=0)

    # Warm-up (JIT of numpy paths / caches).
    for rows, cur in pool[:50]:
        engine.rows = rows
        engine.current = cur
        engine.candidate_features()

    n = len(pool)
    total_cycles = 0
    total_placements = 0
    t0 = time.perf_counter()
    for i in range(args.iters):
        rows, cur = pool[i % n]
        engine.rows = rows
        engine.current = cur
        placements, _feats, _ = engine.candidate_features()
        total_cycles += 1
        total_placements += len(placements)
    elapsed = time.perf_counter() - t0

    cps = total_cycles / elapsed
    avg_pl = total_placements / total_cycles
    print(f"cycles           : {total_cycles}")
    print(f"elapsed_sec      : {elapsed:.4f}")
    print(f"cycles_per_sec   : {cps:.1f}")
    print(f"avg_placements   : {avg_pl:.1f}")
    print(f"gate (>=1000/s)  : {'PASS' if cps >= 1000 else 'FAIL'}")
    return 0 if cps >= 1000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
