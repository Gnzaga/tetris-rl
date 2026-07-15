// Golden parity tests (PLAN.md §5).
//
// For each of the 25 recorded fixtures, run two modes through the JS engine:
//   (1) replay: apply the recorded (rotation, column) moves, asserting the board
//       hash, cumulative lines, and 8 feature values match at every step;
//   (2) re-derive: independently choose moves with the JS Dellacherie agent,
//       asserting the SAME moves plus identical hashes/lines/features.
//
// The board hash is 32-bit FNV-1a over the 40 little-endian bytes of the 20
// uint16 rows (low byte first) — mirrors scripts/gen_fixtures.py exactly.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { makeEngine } from "../demo/js/engine.js";
import { dellacherieAgent } from "../demo/js/agents.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

const piecesJson = JSON.parse(readFileSync(join(ROOT, "shared", "pieces.json"), "utf8"));
const fixtures = JSON.parse(
  readFileSync(join(ROOT, "shared", "fixtures", "parity_v1.json"), "utf8"),
);

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

function round6(x) {
  return Number(x.toFixed(6));
}

assert.equal(fixtures.fixtures.length, 25, "expected 25 fixtures");

for (const fx of fixtures.fixtures) {
  const { seed, moves, hashes, lines, features } = fx;

  test(`seed ${seed} — replay recorded moves`, () => {
    const engine = new TetrisEngine(seed);
    for (let i = 0; i < moves.length; i++) {
      const [r, c] = moves[i];
      const info = engine.step(r, c);
      assert.equal(engine.gameOver && i < moves.length - 1, false, `early game over at step ${i}`);
      assert.equal(fnv1aBoard(engine.rows), hashes[i] >>> 0, `hash mismatch at step ${i}`);
      assert.equal(engine.lines, lines[i], `lines mismatch at step ${i}`);
      const got = info.features.map(round6);
      assert.deepEqual(got, features[i], `feature mismatch at step ${i}`);
    }
  });

  test(`seed ${seed} — re-derive moves with JS Dellacherie`, () => {
    const engine = new TetrisEngine(seed);
    const agent = dellacherieAgent();
    for (let i = 0; i < moves.length; i++) {
      const move = agent.act(engine);
      assert.notEqual(move, null, `no legal move at step ${i}`);
      assert.deepEqual(move, moves[i], `move mismatch at step ${i}`);
      const info = engine.step(move[0], move[1]);
      assert.equal(fnv1aBoard(engine.rows), hashes[i] >>> 0, `hash mismatch at step ${i}`);
      assert.equal(engine.lines, lines[i], `lines mismatch at step ${i}`);
      const got = info.features.map(round6);
      assert.deepEqual(got, features[i], `feature mismatch at step ${i}`);
    }
  });
}
