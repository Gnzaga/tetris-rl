// Agent runner layer for the browser demo (PLAN.md §10).
//
// Wraps the three agent families behind one async `decide(engine)` interface and
// keeps the board->tensor encode, reward, and q-combination logic as pure,
// DOM-free, ORT-free functions so they can be unit-tested under Node. The ORT
// runtime is loaded lazily via dynamic import so importing this module in Node
// (for the pure-logic tests) never touches the browser wasm bundle.

import { Mulberry32 } from "./rng.js";

export const CLEAR_POINTS = [0, 1, 3, 5, 8];
export const GAMMA = 0.95;
// ValueNet parameter count (tetris/model.py): three 3x3 convs + two FCs.
export const VALUENET_PARAMS = 160 + 4640 + 9248 + 819328 + 129; // 833505

function popcount(x) {
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0f0f0f0f;
  return (Math.imul(x, 0x01010101) >> 24) & 0xff;
}

export function boardCells(rows) {
  let n = 0;
  for (const v of rows) n += popcount(v & 0x3ff);
  return n;
}

// (N boards of 20 row-ints) -> Float32Array laid out [N,1,20,10] row-major.
// Cell (row, c) = bit c of the row int (matches tetris/model.py boards_to_array:
// arr[..., c] = (row >> c) & 1). Row 0 = top.
export function encodeBoards(boards) {
  const n = boards.length;
  const out = new Float32Array(n * 200);
  for (let b = 0; b < n; b++) {
    const rows = boards[b];
    const base = b * 200;
    for (let r = 0; r < 20; r++) {
      const v = rows[r];
      if (!v) continue;
      const rb = base + r * 10;
      for (let c = 0; c < 10; c++) {
        if (v & (1 << c)) out[rb + c] = 1.0;
      }
    }
  }
  return out;
}

// Pre-clear decision reward r_i = clear_points[lines_i] with beta = 0 (matching
// Python decision_rewards for the demo/eval path). A placement adds 4 cells and
// each cleared line removes 10, so lines_i = (cells_before + 4 - cells_after)/10.
export function candidateRewards(curRows, afters) {
  const cur = boardCells(curRows);
  const out = new Float32Array(afters.length);
  for (let i = 0; i < afters.length; i++) {
    let lines = Math.floor((cur + 4 - boardCells(afters[i])) / 10);
    if (lines < 0) lines = 0;
    else if (lines > 4) lines = 4;
    out[i] = CLEAR_POINTS[lines];
  }
  return out;
}

// q_i = r_i + gamma * V(A_i).
export function combineQ(r, values, gamma = GAMMA) {
  const out = new Float32Array(r.length);
  for (let i = 0; i < r.length; i++) out[i] = r[i] + gamma * values[i];
  return out;
}

// Argmax with first-maximal tie-break (matches np.argmax / the JS LinearAgent).
export function argmax(arr) {
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > bestVal) {
      bestVal = arr[i];
      best = i;
    }
  }
  return best;
}

export function linearScores(feats, weights) {
  const out = new Array(feats.length);
  for (let i = 0; i < feats.length; i++) {
    const f = feats[i];
    let s = 0;
    for (let k = 0; k < 8; k++) s += f[k] * weights[k];
    out[i] = s;
  }
  return out;
}

// Numerically-stable softmax over an array; returns a plain array.
export function softmax(scores) {
  let m = -Infinity;
  for (const s of scores) if (s > m) m = s;
  let sum = 0;
  const e = new Array(scores.length);
  for (let i = 0; i < scores.length; i++) {
    const x = Math.exp(scores[i] - m);
    e[i] = x;
    sum += x;
  }
  if (sum === 0) return e.map(() => 1 / scores.length);
  for (let i = 0; i < e.length; i++) e[i] /= sum;
  return e;
}

// Indices of the top-k scores, highest first (stable first-maximal tie-break).
export function topKIndices(scores, k) {
  return scores
    .map((s, i) => [s, i])
    .sort((a, b) => (b[0] - a[0]) || (a[1] - b[1]))
    .slice(0, k)
    .map(([, i]) => i);
}

// ---- Runners ---------------------------------------------------------------
// Each decide() returns { placement:[rot,col], placements, scores, afters, chosen }
// or null when the game has no legal move. `scores` is aligned to `placements`
// and drives the heatmap; `afters` (post-clear boards) is present for value nets.

export class RandomRunner {
  constructor(seed = 0) {
    this.rng = new Mulberry32(seed >>> 0);
    this.params = 0;
    this.fileSize = 0;
  }

  async decide(engine) {
    const placements = engine.legalPlacements();
    if (placements.length === 0) return null;
    let idx = Math.floor(this.rng.nextFloat() * placements.length);
    if (idx >= placements.length) idx = placements.length - 1;
    const scores = new Array(placements.length).fill(0);
    scores[idx] = 1;
    return { placement: placements[idx], placements, scores, afters: null, chosen: idx };
  }
}

export class LinearRunner {
  constructor(weights) {
    this.weights = weights.slice();
    this.params = weights.length;
    this.fileSize = 0;
  }

  async decide(engine) {
    const { placements, feats, afters } = engine.candidateFeatures();
    if (placements.length === 0) return null;
    const scores = linearScores(feats, this.weights);
    const chosen = argmax(scores);
    return { placement: placements[chosen], placements, scores, afters, chosen };
  }
}

export class OnnxRunner {
  constructor(ort, session, fileSize = 0) {
    this.ort = ort;
    this.session = session;
    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];
    this.params = VALUENET_PARAMS;
    this.fileSize = fileSize;
  }

  async decide(engine) {
    const { placements, afters } = engine.candidateFeatures();
    if (placements.length === 0) return null;
    const r = candidateRewards(engine.rows, afters);
    const data = encodeBoards(afters);
    const tensor = new this.ort.Tensor("float32", data, [afters.length, 1, 20, 10]);
    const out = await this.session.run({ [this.inputName]: tensor });
    const values = out[this.outputName].data;
    const q = combineQ(r, values, GAMMA);
    const chosen = argmax(q);
    return { placement: placements[chosen], placements, scores: Array.from(q), afters, chosen };
  }
}

// A runner that replays a fixed move list (Replay tab); scores are trivial.
export class ReplayRunner {
  constructor(moves) {
    this.moves = moves;
    this.i = 0;
    this.params = 0;
    this.fileSize = 0;
  }

  async decide(engine) {
    if (this.i >= this.moves.length) return null;
    const move = this.moves[this.i++];
    return { placement: move, placements: [move], scores: [1], afters: null, chosen: 0 };
  }
}

// ---- ORT wiring (browser only) --------------------------------------------

export async function loadOrt() {
  const ort = await import("../vendor/ort.wasm.bundle.min.mjs");
  // Absolute base so ORT resolves the wasm/glue paths unambiguously (a relative
  // wasmPaths is resolved against ORT's own bundle URL and would double the dir).
  ort.env.wasm.wasmPaths = new URL("../vendor/", import.meta.url).href;
  ort.env.wasm.numThreads = 1; // crossOriginIsolated is false on http.server
  ort.env.wasm.proxy = false;
  return ort;
}

async function fetchSize(path) {
  try {
    const resp = await fetch(path, { method: "HEAD" });
    const len = resp.headers.get("content-length");
    return len ? parseInt(len, 10) : 0;
  } catch {
    return 0;
  }
}

// Build a runner for a manifest agent. `ort` may be null for non-onnx agents.
export async function createRunner(agent, ort, seed = 1) {
  switch (agent.type) {
    case "random":
      return new RandomRunner(seed);
    case "linear":
      return new LinearRunner(agent.weights);
    case "onnx": {
      const path = "./models/" + agent.path;
      const [session, size] = await Promise.all([
        ort.InferenceSession.create(path, { executionProviders: ["wasm"] }),
        fetchSize(path),
      ]);
      return new OnnxRunner(ort, session, size);
    }
    default:
      throw new Error(`unknown agent type: ${agent.type}`);
  }
}

// Manifest self-test: run the 3 stored boards through `session` and compare to
// the stored expected values within 1e-3 (PLAN.md §10).
export async function runSelfTest(ort, session, selftest) {
  const boards = selftest.boards;
  const data = encodeBoards(boards);
  const tensor = new ort.Tensor("float32", data, [boards.length, 1, 20, 10]);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const got = Array.from(out[session.outputNames[0]].data);
  const expected = selftest.expected_values;
  let maxErr = 0;
  for (let i = 0; i < expected.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(got[i] - expected[i]));
  }
  return { pass: maxErr <= 1e-3, maxErr, got, expected };
}
