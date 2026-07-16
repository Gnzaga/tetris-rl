// Pure-logic tests for the browser demo runner layer (PLAN.md §10 verify step).
//
// Covers the DOM-free / ORT-free functions that decide what the ONNX and linear
// agents play: board->tensor encode (must match Python boards_to_array), the
// pre-clear reward recovery, q-combination, softmax, argmax tie-break, and
// linear scoring. Also cross-checks encode + reward against a real engine board.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  encodeBoards,
  candidateRewards,
  combineQ,
  argmax,
  softmax,
  topKIndices,
  linearScores,
  boardCells,
  CLEAR_POINTS,
} from "../demo/js/runner.js";
import { makeEngine } from "../demo/js/engine.js";
import { dellacherieAgent, DELLACHERIE_WEIGHTS } from "../demo/js/agents.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const piecesJson = JSON.parse(readFileSync(join(ROOT, "shared", "pieces.json"), "utf8"));
const { TetrisEngine } = makeEngine(piecesJson);

test("encodeBoards: bit c of row -> cell (row, c), row-major [N,1,20,10]", () => {
  // Row value 0b0000000101 = columns 0 and 2 filled; put it on row 3.
  const rows = new Array(20).fill(0);
  rows[3] = 0b0000000101;
  rows[19] = (1 << 9); // column 9 on bottom row
  const out = encodeBoards([rows]);
  assert.equal(out.length, 200);
  assert.equal(out[3 * 10 + 0], 1); // row 3 col 0
  assert.equal(out[3 * 10 + 1], 0); // row 3 col 1
  assert.equal(out[3 * 10 + 2], 1); // row 3 col 2
  assert.equal(out[19 * 10 + 9], 1); // row 19 col 9
  // Everything else zero.
  let ones = 0;
  for (const v of out) if (v) ones++;
  assert.equal(ones, 3);
});

test("encodeBoards: batch layout is contiguous per board", () => {
  const a = new Array(20).fill(0);
  a[0] = 1;
  const b = new Array(20).fill(0);
  b[0] = 2;
  const out = encodeBoards([a, b]);
  assert.equal(out.length, 400);
  assert.equal(out[0], 1); // board 0, row 0, col 0
  assert.equal(out[200 + 1], 1); // board 1, row 0, col 1
  assert.equal(out[200 + 0], 0);
});

test("boardCells counts filled cells (10-bit rows)", () => {
  const rows = new Array(20).fill(0);
  rows[0] = 0x3ff; // full row = 10 cells
  rows[5] = 0b101; // 2 cells
  assert.equal(boardCells(rows), 12);
});

test("candidateRewards: r_i = clear_points[lines], lines from cell delta", () => {
  // cur has one nearly-full row (9 cells in row 19). A placement that fills the
  // 10th clears 1 line: after = all empty. cur_cells=9, +4 piece = 13,
  // after_cells=3 (the other 3 piece cells land elsewhere / cleared)... instead
  // construct precisely: cur=9 cells, after board has 3 cells => lines=(9+4-3)/10=1.
  const cur = new Array(20).fill(0);
  cur[19] = 0x1ff; // 9 cells
  const afterClear = new Array(20).fill(0);
  afterClear[19] = 0b111; // 3 residual cells after the clear
  const afterNoClear = new Array(20).fill(0);
  afterNoClear[19] = 0x1ff;
  afterNoClear[18] = 0b1111; // +4 cells, no clear -> 13 cells
  const r = candidateRewards(cur, [afterClear, afterNoClear]);
  assert.equal(r[0], CLEAR_POINTS[1]); // 1
  assert.equal(r[1], CLEAR_POINTS[0]); // 0
});

test("combineQ = r + gamma*V", () => {
  const q = combineQ(new Float32Array([1, 0]), new Float32Array([2, -1]), 0.95);
  assert.ok(Math.abs(q[0] - (1 + 0.95 * 2)) < 1e-6);
  assert.ok(Math.abs(q[1] - (0 + 0.95 * -1)) < 1e-6);
});

test("argmax picks first maximal index (tie-break)", () => {
  assert.equal(argmax([1, 3, 3, 2]), 1);
  assert.equal(argmax([-5]), 0);
});

test("softmax normalizes and preserves ordering", () => {
  const p = softmax([2, 1, 0]);
  const sum = p.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9);
  assert.ok(p[0] > p[1] && p[1] > p[2]);
  // Stable under large values.
  const p2 = softmax([1000, 1000]);
  assert.ok(Math.abs(p2[0] - 0.5) < 1e-9);
});

test("topKIndices returns highest-first, stable ties", () => {
  assert.deepEqual(topKIndices([1, 5, 3, 5], 2), [1, 3]);
  assert.deepEqual(topKIndices([9], 5), [0]);
});

test("linearScores matches manual dot product and the JS LinearAgent choice", () => {
  const feats = [
    [1, 0, 2, 3, 0, 1, 0, 0],
    [0, 2, 0, 1, 1, 0, 0, 0],
  ];
  const s = linearScores(feats, DELLACHERIE_WEIGHTS);
  // manual: w=[-1,1,-1,-1,-4,-1,0,0]
  assert.equal(s[0], -1 * 1 + 0 - 1 * 2 - 1 * 3 - 0 - 1 * 1); // -7
  assert.equal(s[1], 0 + 1 * 2 - 0 - 1 * 1 - 4 * 1 - 0); // -3
});

test("linear runner logic reproduces engine candidateFeatures argmax", () => {
  // The demo's LinearRunner path (candidateFeatures + linearScores + argmax) must
  // pick the same placement the parity-locked LinearAgent picks.
  const engine = new TetrisEngine(12345);
  const agent = dellacherieAgent();
  for (let i = 0; i < 40 && !engine.gameOver; i++) {
    const { placements, feats } = engine.candidateFeatures();
    if (!placements.length) break;
    const scores = linearScores(feats, DELLACHERIE_WEIGHTS);
    const chosen = argmax(scores);
    const agentMove = agent.act(engine);
    assert.deepEqual(placements[chosen], agentMove);
    engine.step(...agentMove);
  }
});

test("candidateRewards matches lines actually cleared by the engine", () => {
  const engine = new TetrisEngine(777);
  for (let i = 0; i < 60 && !engine.gameOver; i++) {
    const { placements, afters } = engine.candidateFeatures();
    if (!placements.length) break;
    const r = candidateRewards(engine.rows, afters);
    // Pick a random legal placement, step, and confirm r matched its clear.
    const j = i % placements.length;
    const expected = CLEAR_POINTS[r[j] === 0 ? 0 : CLEAR_POINTS.indexOf(r[j])];
    const info = engine.step(...placements[j]);
    assert.equal(r[j], CLEAR_POINTS[info.linesCleared]);
    assert.equal(expected, r[j]);
  }
});
