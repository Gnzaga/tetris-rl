"""Generate v2 parity fixtures (PLAN2.md §3/§5, gen_fixtures_v2 parts 1 & 2).

Part 1 (frame-layer parity, shared/fixtures/parity_v2.json)
-----------------------------------------------------------
For each of NUM_SEEDS seeds, drive the frame env for NUM_TICKS ticks with a
seeded pseudo-random action sequence and record, at every decision tick, the
tick index, a 32-bit FNV-1a board hash (v1 scheme), and the active piece pose
(piece id, rot, col, row) plus the gravity counter; also record every lock event
with its (rotation, column), the cumulative lines after the lock, and a tuck
flag (1 if the piece locked deeper than the v1 straight drop of its (rot, col)
— physical-pose locking, see tetris/frame_env.py). The JS suite
(tests_js/parity_v2.test.mjs) regenerates the same action stream from the seed,
replays it through the JS frame env, asserts every recorded field matches
bit-for-bit, and cross-checks the v1-consistency invariant against a parallel
bare v1 engine over the non-tuck prefix of each game.

Part 2 (observation parity, shared/fixtures/obs_v2.json)
--------------------------------------------------------
For each of OBS_NUM_SEEDS seeds, replay the SAME seeded action stream and, at the
first OBS_NUM_DECISIONS decision ticks, render the 96x96 observation
(tetris/render_obs.py) that the agent would see at that tick (before applying the
tick's action) and record the CRC32 (zlib/IEEE) of its 9216 row-major bytes. The
JS suite renders with demo/js/render_obs.js and asserts each CRC32 matches
bit-exactly.

Action scheme (frozen, regenerable in JS): a separate Mulberry32 seeded with the
fixture seed; at each decision tick, action = floor(next_float() * 5), mapped to
[noop, left, right, rot_cw, rot_ccw].

Board hash: identical to gen_fixtures.py — 32-bit FNV-1a (offset 0x811c9dc5,
prime 0x01000193) over the 40 little-endian bytes of the 20 uint16 rows.

Outputs: shared/fixtures/parity_v2.json, shared/fixtures/obs_v2.json.
"""

import _pathshim  # noqa: F401
import argparse
import json
import zlib
from pathlib import Path

from tetris.frame_env import DECISION_PERIOD, GRAVITY_PERIOD, FrameEnv
from tetris.render_obs import (
    BORDER_X0,
    BORDER_X1,
    BORDER_Y0,
    BORDER_Y1,
    OBS_SIZE,
    PREVIEW_X,
    PREVIEW_Y,
    render_env,
)
from tetris.rng import Mulberry32

NUM_SEEDS = 15
NUM_TICKS = 3000
NUM_ACTIONS = 5
ENGINE_VERSION = "1"

# Part 2 (observation fixtures).
OBS_NUM_SEEDS = 5
OBS_NUM_DECISIONS = 50

_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193
_U32 = 0xFFFFFFFF

_FIX_DIR = Path(__file__).resolve().parent.parent / "shared" / "fixtures"
_OUT_PATH = _FIX_DIR / "parity_v2.json"
_OBS_OUT_PATH = _FIX_DIR / "obs_v2.json"


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
            "score": int(env.score),
            "game_over": bool(env.game_over),
            "ticks": NUM_TICKS,
        },
    }


def obs_fixture(seed: int) -> dict:
    """Replay the seeded action stream, recording the CRC32 of the first
    OBS_NUM_DECISIONS observations (rendered at each decision tick, before that
    tick's action is applied)."""
    env = FrameEnv(seed=seed)
    action_rng = Mulberry32(seed)
    crcs: list[int] = []
    while len(crcs) < OBS_NUM_DECISIONS and not env.game_over:
        if env.tick_count % DECISION_PERIOD == 0:
            # Render what the agent observes at this decision tick, then act.
            obs = render_env(env)
            crcs.append(zlib.crc32(obs.tobytes()) & _U32)
            env.apply_action(int(action_rng.next_float() * NUM_ACTIONS))
        env.tick()
    return {"seed": seed, "crcs": crcs}


def write_parity() -> int:
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


def write_obs() -> int:
    fixtures = [obs_fixture(seed) for seed in range(OBS_NUM_SEEDS)]
    payload = {
        "engine_version": ENGINE_VERSION,
        "crc": "zlib.crc32 (CRC-32/IEEE) over the 9216 row-major bytes of the 96x96 uint8 obs",
        "obs_size": OBS_SIZE,
        "num_seeds": OBS_NUM_SEEDS,
        "num_decisions": OBS_NUM_DECISIONS,
        "decision_period": DECISION_PERIOD,
        "num_actions": NUM_ACTIONS,
        "action_scheme": (
            "separate Mulberry32(seed); at each decision tick action = "
            "floor(next_float() * 5); the obs is rendered BEFORE that tick's action"
        ),
        "layout": {
            "border_rect_inclusive": [BORDER_X0, BORDER_Y0, BORDER_X1, BORDER_Y1],
            "board_topleft": [8, 8],
            "cell_px": 4,
            "preview_topleft": [PREVIEW_X, PREVIEW_Y],
            "preview_px": 20,
        },
        "fixtures": fixtures,
    }
    _OBS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OBS_OUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    total = sum(len(fx["crcs"]) for fx in fixtures)
    size = _OBS_OUT_PATH.stat().st_size
    print(f"wrote {_OBS_OUT_PATH} : {len(fixtures)} fixtures, {total} obs CRCs, {size} bytes")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate v2 parity fixtures (PLAN2.md §3/§5)")
    ap.add_argument("--part", choices=["1", "2", "all"], default="all",
                    help="1=frame parity, 2=obs parity, all=both (default)")
    args = ap.parse_args(argv)
    if args.part in ("1", "all"):
        write_parity()
    if args.part in ("2", "all"):
        write_obs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
