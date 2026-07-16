"""Generate v2 frame-layer parity fixtures (PLAN2.md §3, gen_fixtures_v2 part 1).

For each of NUM_SEEDS seeds, drive the frame env for NUM_TICKS ticks with a
seeded pseudo-random action sequence and record, at every decision tick, the
tick index, a 32-bit FNV-1a board hash (v1 scheme), and the active piece pose
(piece id, rot, col, row) plus the gravity counter; also record every lock event
with its derived (rotation, column), the cumulative lines after the lock, and a
tuck flag (1 if the piece rested deeper than a v1 straight drop — see
tetris/frame_env.py). The JS suite (tests_js/parity_v2.test.mjs) regenerates the
same action stream from the seed, replays it through the JS frame env, and
asserts every recorded field matches bit-for-bit.

Action scheme (frozen, regenerable in JS): a separate Mulberry32 seeded with the
fixture seed; at each decision tick, action = floor(next_float() * 5), mapped to
[noop, left, right, rot_cw, rot_ccw].

Board hash: identical to gen_fixtures.py — 32-bit FNV-1a (offset 0x811c9dc5,
prime 0x01000193) over the 40 little-endian bytes of the 20 uint16 rows.

Output: shared/fixtures/parity_v2.json.
"""

import _pathshim  # noqa: F401
import json
from pathlib import Path

from tetris.frame_env import DECISION_PERIOD, GRAVITY_PERIOD, FrameEnv
from tetris.rng import Mulberry32

NUM_SEEDS = 15
NUM_TICKS = 3000
NUM_ACTIONS = 5
ENGINE_VERSION = "1"

_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193
_U32 = 0xFFFFFFFF

_OUT_PATH = Path(__file__).resolve().parent.parent / "shared" / "fixtures" / "parity_v2.json"


def fnv1a_board(rows) -> int:
    """32-bit FNV-1a over 40 little-endian bytes of the 20 uint16 rows."""
    h = _FNV_OFFSET
    for v in rows:
        for b in (v & 0xFF, (v >> 8) & 0xFF):
            h = ((h ^ b) * _FNV_PRIME) & _U32
    return h


def play_fixture(seed: int) -> dict:
    env = FrameEnv(seed=seed)
    action_rng = Mulberry32(seed)
    decisions = []
    locks = []
    for t in range(NUM_TICKS):
        if env.tick_count % DECISION_PERIOD == 0:
            action = int(action_rng.next_float() * NUM_ACTIONS)
            env.apply_action(action)
        lock = env.tick()
        if lock is not None:
            locks.append(
                [lock["tick"], lock["r"], lock["c"], lock["lines_after"], int(lock["tuck"])]
            )
        if t % DECISION_PERIOD == 0:
            decisions.append(
                [t, fnv1a_board(env.rows), env.piece, env.rot, env.col, env.row, env.gravity_counter]
            )
    return {
        "seed": seed,
        "decisions": decisions,
        "locks": locks,
        "final": {
            "lines": int(env.lines),
            "pieces": int(env.pieces),
            "game_over": bool(env.game_over),
            "ticks": NUM_TICKS,
        },
    }


def main() -> int:
    fixtures = [play_fixture(seed) for seed in range(NUM_SEEDS)]
    payload = {
        "engine_version": ENGINE_VERSION,
        "hash": "fnv1a-32 over 40 little-endian bytes of the 20 uint16 rows (low byte first)",
        "num_seeds": NUM_SEEDS,
        "num_ticks": NUM_TICKS,
        "decision_period": DECISION_PERIOD,
        "gravity_period": GRAVITY_PERIOD,
        "num_actions": NUM_ACTIONS,
        "action_scheme": (
            "separate Mulberry32(seed); at each decision tick action = "
            "floor(next_float() * 5) => [noop, left, right, rot_cw, rot_ccw]"
        ),
        "decision_fields": ["tick", "board_hash", "piece", "rot", "col", "row", "gravity_counter"],
        "lock_fields": ["tick", "r", "c", "lines_after", "tuck"],
        "fixtures": fixtures,
    }
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    total_dec = sum(len(fx["decisions"]) for fx in fixtures)
    total_lock = sum(len(fx["locks"]) for fx in fixtures)
    total_tuck = sum(lk[4] for fx in fixtures for lk in fx["locks"])
    size = _OUT_PATH.stat().st_size
    print(
        f"wrote {_OUT_PATH} : {len(fixtures)} fixtures, {total_dec} decisions, "
        f"{total_lock} locks ({total_tuck} tucks), {size} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
