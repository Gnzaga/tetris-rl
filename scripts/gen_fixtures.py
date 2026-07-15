"""Generate golden parity fixtures (PLAN.md §5).

For each of 25 seeds, play a full Dellacherie-driven game capped at 500 pieces
and record, after every step: the chosen (rotation, column), a 32-bit FNV-1a
board hash, cumulative lines_cleared, and the 8 feature values of the chosen
placement (rounded to 6 decimals). The JS parity suite replays these AND
independently re-derives the moves with the JS Dellacherie agent, asserting
identical hashes/lines/features at every step.

Board hash: 32-bit FNV-1a (offset basis 0x811c9dc5, prime 0x01000193) over the
40 bytes formed by emitting each of the 20 row ints as 2 little-endian bytes
(low byte first): for row v, feed (v & 0xFF) then ((v >> 8) & 0xFF). This exact
byte order is mirrored in tests_js/parity.test.mjs.

Output: shared/fixtures/parity_v1.json.
"""

import _pathshim  # noqa: F401
import json
from pathlib import Path

from tetris.agents import dellacherie_agent
from tetris.engine import TetrisEngine

NUM_SEEDS = 25
MAX_PIECES = 500
ENGINE_VERSION = "1"

_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193
_U32 = 0xFFFFFFFF

_OUT_PATH = Path(__file__).resolve().parent.parent / "shared" / "fixtures" / "parity_v1.json"


def fnv1a_board(rows) -> int:
    """32-bit FNV-1a over 40 little-endian bytes of the 20 uint16 rows."""
    h = _FNV_OFFSET
    for v in rows:
        for b in (v & 0xFF, (v >> 8) & 0xFF):
            h = ((h ^ b) * _FNV_PRIME) & _U32
    return h


def round6(x) -> float:
    return round(float(x), 6)


def play_fixture(seed: int) -> dict:
    engine = TetrisEngine(seed=seed)
    agent = dellacherie_agent()
    moves = []
    hashes = []
    lines = []
    features = []
    while not engine.game_over and engine.pieces < MAX_PIECES:
        move = agent.act(engine)
        if move is None:
            break
        info = engine.step(*move)
        moves.append([int(move[0]), int(move[1])])
        hashes.append(fnv1a_board(engine.rows))
        lines.append(int(engine.lines))
        features.append([round6(x) for x in info.features])
    return {
        "seed": seed,
        "moves": moves,
        "hashes": hashes,
        "lines": lines,
        "features": features,
    }


def main() -> int:
    fixtures = [play_fixture(seed) for seed in range(NUM_SEEDS)]
    payload = {
        "engine_version": ENGINE_VERSION,
        "hash": "fnv1a-32 over 40 little-endian bytes of the 20 uint16 rows (low byte first)",
        "max_pieces": MAX_PIECES,
        "fixtures": fixtures,
    }
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    total = sum(len(fx["moves"]) for fx in fixtures)
    size = _OUT_PATH.stat().st_size
    print(f"wrote {_OUT_PATH} : {len(fixtures)} fixtures, {total} steps, {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
