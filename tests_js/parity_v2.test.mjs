// Frame-layer golden parity tests (PLAN2.md §3, Phase A gate).
//
// For each of the 15 recorded v2 fixtures, regenerate the seeded action stream,
// replay it through the JS frame env (physical-pose locking) for the recorded
// number of ticks, and assert every recorded field matches the Python fixture
// bit-for-bit:
//   * at every decision tick: tick index, board hash, piece id, rot, col, row,
//     gravity counter;
//   * every lock event: tick, (rotation, column), cumulative lines, tuck flag;
//   * final lines / pieces / score / game_over.
//
// v1-consistency invariant (amended §1): in parallel, a bare v1 TetrisEngine is
// stepped with each lock's (r, c) and its board/lines compared after every lock
// over the NON-TUCK PREFIX of the game — every non-tuck lock transition must be
// bit-identical to v1 engine.step. The cross-check stops at the first tuck
// (boards legitimately diverge after a physical tuck lock).
//
// A hand-built tuck scenario asserts the exact post-lock board when a piece
// slides under an overhang and locks below its straight-drop row.
//
// The action stream uses a separate Mulberry32 seeded with the fixture seed:
// action = floor(nextFloat() * 5) at each decision tick. The board hash is the
// v1 32-bit FNV-1a over the 40 little-endian bytes of the 20 uint16 rows.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { makeEngine } from "../demo/js/engine.js";
import {
  makeFrameEnv,
  DECISION_PERIOD,
  GRAVITY_PERIOD,
  RIGHT,
} from "../demo/js/frame_env.js";
import { HEIGHT } from "../demo/js/features.js";
import { Mulberry32 } from "../demo/js/rng.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

const piecesJson = JSON.parse(readFileSync(join(ROOT, "shared", "pieces.json"), "utf8"));
const fixtures = JSON.parse(
  readFileSync(join(ROOT, "shared", "fixtures", "parity_v2.json"), "utf8"),
);

const { FrameEnv } = makeFrameEnv(piecesJson);
const { TetrisEngine } = makeEngine(piecesJson);

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

    // Parallel bare v1 engine for the v1-consistency cross-check (non-tuck
    // prefix only; boards legitimately diverge after the first tuck lock).
    const bare = new TetrisEngine(seed);
    let crossCheck = true;

    for (let t = 0; t < NUM_TICKS; t++) {
      if (env.tickCount % DECISION_PERIOD === 0) {
        env.apply_action(Math.floor(actionRng.nextFloat() * NUM_ACTIONS));
      }
      const lock = env.tick();
      if (lock !== null) {
        gotLocks.push([lock.tick, lock.r, lock.c, lock.lines_after, lock.tuck ? 1 : 0]);
        if (crossCheck) {
          if (lock.tuck) {
            crossCheck = false; // physical tuck: v1 comparison ends here
          } else if (!bare.gameOver) {
            bare.step(lock.r, lock.c);
            assert.deepEqual(
              env.rows,
              bare.rows,
              `v1-consistency board mismatch @tick ${lock.tick}`,
            );
            assert.equal(
              env.lines,
              bare.lines,
              `v1-consistency lines mismatch @tick ${lock.tick}`,
            );
            // v1 may end earlier (its "new piece has no legal placement" rule
            // does not exist in the frame layer) — stop cross-checking then.
            if (bare.gameOver) crossCheck = false;
          }
        }
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
    assert.equal(env.score, final.score, "final score mismatch");
    assert.equal(env.gameOver, final.game_over, "final game_over mismatch");
  });
}

test("hand-built tuck locks at the physical pose (exact post-lock board)", () => {
  // Mirror of tests/test_frame_env.py::test_constructed_tuck. A vertical I in
  // open column 0 slides right under an overhang (col 1 filled at row 10 only)
  // and locks at rows 16..19 of column 1 — below the straight-drop rest
  // (rows 6..9). Physical-pose locking keeps the cells where the camera saw
  // them; the overhang cell remains at row 10.
  const env = new FrameEnv(0);
  env.piece = 0; // I
  env._spawn();
  env.rows = new Array(HEIGHT).fill(0);
  env.rows[10] = 1 << 1; // overhang: col 1 filled at row 10, empty below
  env.rot = 1; // vertical I, width 1
  env.col = 0;
  env.row = 15; // cells rows 15..18 in col 0, below the overhang

  assert.equal(env._straightDropRow(1, 1), 6, "straight drop should rest on the overhang");

  env.tickCount = 0; // force a decision tick
  env.gravityCounter = 0;
  env.apply_action(RIGHT);
  env.tick();
  assert.equal(env.col, 1, "slide under the overhang should succeed");
  assert.equal(env.row, 15);

  // One more gravity descent takes it to rows 16..19 (floor), then it locks.
  let lock = null;
  for (let i = 0; i < GRAVITY_PERIOD * 6; i++) {
    const lk = env.tick();
    if (lk !== null) {
      lock = lk;
      break;
    }
  }
  assert.notEqual(lock, null, "piece should lock");
  assert.equal(lock.r, 1);
  assert.equal(lock.c, 1);
  assert.equal(lock.tuck, true);
  assert.equal(env.gameOver, false);

  // Exact post-lock board: col 1 filled at rows 10 (overhang) and 16..19.
  const expected = new Array(HEIGHT).fill(0);
  for (const r of [10, 16, 17, 18, 19]) expected[r] = 1 << 1;
  assert.deepEqual(env.rows, expected, "tuck must lock at the physical pose");
});
