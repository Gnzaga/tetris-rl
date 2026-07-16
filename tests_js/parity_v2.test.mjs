// Frame-layer golden parity tests (PLAN2.md §3, Phase A gate).
//
// For each of the 15 recorded v2 fixtures, regenerate the seeded action stream,
// replay it through the JS frame env for the recorded number of ticks, and
// assert every recorded field matches the Python fixture bit-for-bit:
//   * at every decision tick: tick index, board hash, piece id, rot, col, row,
//     gravity counter;
//   * every lock event: tick, derived (rotation, column), cumulative lines, and
//     the tuck flag;
//   * final lines / pieces / game_over.
//
// The action stream uses a separate Mulberry32 seeded with the fixture seed:
// action = floor(nextFloat() * 5) at each decision tick. The board hash is the
// v1 32-bit FNV-1a over the 40 little-endian bytes of the 20 uint16 rows.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { makeFrameEnv, DECISION_PERIOD } from "../demo/js/frame_env.js";
import { Mulberry32 } from "../demo/js/rng.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

const piecesJson = JSON.parse(readFileSync(join(ROOT, "shared", "pieces.json"), "utf8"));
const fixtures = JSON.parse(
  readFileSync(join(ROOT, "shared", "fixtures", "parity_v2.json"), "utf8"),
);

const { FrameEnv } = makeFrameEnv(piecesJson);

const FNV_OFFSET = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

function fnv1aBoard(rows) {
  let h = FNV_OFFSET >>> 0;
  for (const v of rows) {
    for (const b of [v & 0xff, (v >> 8) & 0xff]) {
      h = Math.imul((h ^ b) >>> 0, FNV_PRIME) >>> 0;
    }
  }
  return h >>> 0;
}

assert.equal(fixtures.fixtures.length, 15, "expected 15 fixtures");
assert.equal(fixtures.decision_period, DECISION_PERIOD, "decision period mismatch");

const NUM_ACTIONS = fixtures.num_actions;
const NUM_TICKS = fixtures.num_ticks;

for (const fx of fixtures.fixtures) {
  const { seed, decisions, locks, final } = fx;

  test(`seed ${seed} — replay seeded action stream`, () => {
    const env = new FrameEnv(seed);
    const actionRng = new Mulberry32(seed);
    const gotDecisions = [];
    const gotLocks = [];

    for (let t = 0; t < NUM_TICKS; t++) {
      if (env.tickCount % DECISION_PERIOD === 0) {
        env.apply_action(Math.floor(actionRng.nextFloat() * NUM_ACTIONS));
      }
      const lock = env.tick();
      if (lock !== null) {
        gotLocks.push([lock.tick, lock.r, lock.c, lock.lines_after, lock.tuck ? 1 : 0]);
      }
      if (t % DECISION_PERIOD === 0) {
        gotDecisions.push([
          t,
          fnv1aBoard(env.rows),
          env.piece,
          env.rot,
          env.col,
          env.row,
          env.gravityCounter,
        ]);
      }
    }

    assert.equal(gotDecisions.length, decisions.length, "decision count mismatch");
    for (let i = 0; i < decisions.length; i++) {
      const exp = decisions[i].slice();
      const got = gotDecisions[i].slice();
      // hashes are recorded unsigned; normalize both sides to >>> 0.
      exp[1] = exp[1] >>> 0;
      got[1] = got[1] >>> 0;
      assert.deepEqual(got, exp, `decision mismatch at index ${i} (tick ${decisions[i][0]})`);
    }

    assert.equal(gotLocks.length, locks.length, "lock count mismatch");
    for (let i = 0; i < locks.length; i++) {
      assert.deepEqual(gotLocks[i], locks[i], `lock mismatch at index ${i}`);
    }

    assert.equal(env.lines, final.lines, "final lines mismatch");
    assert.equal(env.pieces, final.pieces, "final pieces mismatch");
    assert.equal(env.gameOver, final.game_over, "final game_over mismatch");
  });
}
