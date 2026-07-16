// Pure-logic tests for the pixel agent (PLAN2.md §8 verify step).
//
// Covers the DOM-free / ORT-free helpers that decide what the pixel agent plays:
// obs 4-stack reconstruction order + normalization (must match tetris/bc.py),
// argmax->action, the action->keypad map, and pixel_agents manifest parsing
// (including that a v1-only manifest yields null so the v1 demo is untouched).

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  argmaxLogits,
  stackToTensor,
  pushHistory,
  parsePixelManifest,
  ACTION_LEGEND,
  ACTION_NAMES,
  ACTION_TO_KEY,
  STACK,
} from "../demo/js/pixel_agent.js";
import { NOOP, LEFT, RIGHT, ROT_CW, ROT_CCW } from "../demo/js/frame_env.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const FRAME = 96 * 96;

test("argmaxLogits: first-maximal tie-break (matches numpy argmax)", () => {
  assert.equal(argmaxLogits([0.1, 0.9, 0.2, 0.9, 0.3]), 1);
  assert.equal(argmaxLogits([-5, -1, -9]), 1);
  assert.equal(argmaxLogits([2]), 0);
});

test("pushHistory: empty fills 4x; then append-and-drop-oldest (deque maxlen=4)", () => {
  const o0 = new Uint8Array(FRAME).fill(1);
  const o1 = new Uint8Array(FRAME).fill(2);
  const o2 = new Uint8Array(FRAME).fill(3);

  let h = pushHistory([], o0);
  assert.equal(h.length, 4);
  assert.ok(h.every((f) => f === o0), "first push repeats the current obs 4x");

  h = pushHistory(h, o1);
  assert.deepEqual(h.map((f) => f[0]), [1, 1, 1, 2], "oldest->newest, oldest dropped");
  h = pushHistory(h, o2);
  assert.deepEqual(h.map((f) => f[0]), [1, 1, 2, 3]);
});

test("stackToTensor: channel-major [4,96,96], oldest first, values /255", () => {
  const frames = [0, 128, 255, 64].map((v) => new Uint8Array(FRAME).fill(v));
  const t = stackToTensor(frames);
  assert.equal(t.length, STACK * FRAME);
  // Channel c occupies [c*FRAME .. (c+1)*FRAME); value = raw/255.
  // Float32Array rounds to fp32, so compare within fp32 precision.
  assert.ok(Math.abs(t[0 * FRAME] - 0 / 255) < 1e-6);
  assert.ok(Math.abs(t[1 * FRAME] - 128 / 255) < 1e-6);
  assert.ok(Math.abs(t[2 * FRAME] - 255 / 255) < 1e-6);
  assert.ok(Math.abs(t[3 * FRAME] - 64 / 255) < 1e-6);
});

test("action legend / names / keypad map are aligned and complete", () => {
  assert.equal(ACTION_LEGEND.length, 5);
  assert.equal(ACTION_NAMES.length, 5);
  assert.deepEqual(ACTION_NAMES, ["noop", "left", "right", "rot_cw", "rot_ccw"]);
  // Rotations -> up/down, slides -> left/right, noop -> centre dot.
  assert.equal(ACTION_TO_KEY[NOOP], "noop");
  assert.equal(ACTION_TO_KEY[LEFT], "left");
  assert.equal(ACTION_TO_KEY[RIGHT], "right");
  assert.equal(ACTION_TO_KEY[ROT_CW], "up");
  assert.equal(ACTION_TO_KEY[ROT_CCW], "down");
});

test("parsePixelManifest: v1-only manifest -> null (v1 demo untouched)", () => {
  assert.equal(parsePixelManifest({ agents: [], engine_version: "1" }), null);
  assert.equal(parsePixelManifest({ pixel_agents: { agents: [] } }), null);
  assert.equal(parsePixelManifest({}), null);
});

test("parsePixelManifest: real demo manifest exposes v2 pixel block", () => {
  const manifest = JSON.parse(
    readFileSync(join(ROOT, "demo", "models", "manifest.json"), "utf8"),
  );
  // v1 keys still present (demo compatibility).
  assert.ok(Array.isArray(manifest.agents), "v1 agents list intact");
  assert.ok(manifest.selftest, "v1 selftest intact");

  const pm = parsePixelManifest(manifest);
  assert.ok(pm, "pixel_agents block parsed");
  assert.ok(pm.agents.length >= 5, "milestone progression present");
  assert.ok(pm.byId[pm.final], "final id resolves to an agent");
  assert.equal(pm.inputName, "obs");
  assert.deepEqual(pm.outputNames.slice(0, 5), ["logits", "conv1", "conv2", "conv3", "fc"]);
  assert.deepEqual(pm.activationOutputs, ["conv1", "conv2", "conv3", "fc"]);
  assert.deepEqual(pm.legend, ACTION_LEGEND);
  // FC->action weight matrix for the wire graph: 5x256.
  assert.equal(pm.fcWeight.length, 5);
  assert.equal(pm.fcWeight[0].length, 256);
  assert.equal(pm.fcBias.length, 5);
  // Self-test sidecar reference.
  assert.ok(pm.selftest && pm.selftest.path, "selftest_pixel path present");
  // Every pixel agent entry is well-formed.
  for (const a of pm.agents) {
    assert.equal(typeof a.id, "string");
    assert.equal(typeof a.label, "string");
    assert.ok(a.path.endsWith(".onnx"));
    assert.ok(a.eval && "median_lines" in a.eval);
  }
});
