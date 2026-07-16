// Pixel-agent inference + real-time controller (PLAN2.md §8, Phase F).
//
// The pixel agent sees ONLY the 96×96 rendered observation (render_obs.js) and
// acts ONLY by pressing an arrow key into the frame layer (frame_env.js), in
// proportional real time. This module owns:
//   * pure, DOM-free / ORT-free helpers (obs stacking, argmax->action, manifest
//     parsing) — unit-tested under Node in tests_js/pixel_agent.test.mjs;
//   * PixelAgent: wraps a multi-output ONNX session (logits + conv/fc/aux taps);
//   * PixelController: the loop-facing driver, mirroring controller.js's
//     deciding/dead/tick surface so main.js's 30 Hz accumulator can run it.
//
// Obs stack convention — MUST match tetris/bc.py exactly: the last 4 CONSECUTIVE
// decision-tick observations, oldest→newest [t-3, t-2, t-1, t], normalized /255;
// at episode start the history is padded by repeating the first frame 4×.

import { OBS_SIZE } from "./render_obs.js";
import {
  NOOP, LEFT, RIGHT, ROT_CW, ROT_CCW,
} from "./frame_env.js";

export const ACTION_LEGEND = ["noop", "←", "→", "↑CW", "↓CCW"];
export const ACTION_NAMES = ["noop", "left", "right", "rot_cw", "rot_ccw"];
export const STACK = 4;
const FRAME = OBS_SIZE * OBS_SIZE; // 9216

// Keypad symbol emitted for each of the 5 frame-env actions (emulator arrow pad):
// rotations map to ↑ (cw) / ↓ (ccw), slides to ← / →, noop to the centre dot.
export const ACTION_TO_KEY = {
  [NOOP]: "noop",
  [LEFT]: "left",
  [RIGHT]: "right",
  [ROT_CW]: "up",
  [ROT_CCW]: "down",
};

// argmax with first-maximal tie-break — matches numpy argmax / the Python
// greedy policy (bc.py `_greedy_actions`).
export function argmaxLogits(logits) {
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > bestVal) {
      bestVal = logits[i];
      best = i;
    }
  }
  return best;
}

// frames: array of exactly STACK Uint8Array(FRAME), oldest→newest. Returns a
// Float32Array(STACK*FRAME) laid out [4,96,96] row-major, values x/255. This is
// the exact policy input contract (bc.py batch_stacks / evaluate_policy).
export function stackToTensor(frames) {
  const out = new Float32Array(STACK * FRAME);
  for (let f = 0; f < STACK; f++) {
    const src = frames[f];
    const base = f * FRAME;
    for (let i = 0; i < FRAME; i++) out[base + i] = src[i] / 255;
  }
  return out;
}

// Maintain the rolling 4-history like Python's deque(maxlen=4): the first push
// (empty history) fills all 4 slots with the current obs; later pushes append
// and drop the oldest. Returns the new history array (length 4).
export function pushHistory(history, obs) {
  if (history.length === 0) return [obs, obs, obs, obs];
  const next = history.slice(1);
  next.push(obs);
  return next;
}

// Pull the pixel_agents block out of a manifest into a normalized shape, or null
// if the manifest is v1-only (keeps the v1 demo untouched when absent).
export function parsePixelManifest(manifest) {
  const pa = manifest && manifest.pixel_agents;
  if (!pa || !Array.isArray(pa.agents) || pa.agents.length === 0) return null;
  return {
    agents: pa.agents,
    byId: Object.fromEntries(pa.agents.map((a) => [a.id, a])),
    final: pa.final,
    inputName: pa.input_name || "obs",
    outputNames: pa.output_names || ["logits", "conv1", "conv2", "conv3", "fc"],
    activationOutputs: pa.activation_outputs || ["conv1", "conv2", "conv3", "fc"],
    legend: pa.action_legend || ACTION_LEGEND,
    actionNames: pa.action_names || ACTION_NAMES,
    obsSpec: pa.obs_spec || {},
    fcWeight: pa.fc_action_weight || null,
    fcBias: pa.fc_action_bias || null,
    selftest: pa.selftest_pixel || null,
  };
}

// ---- ORT-backed inference (browser) ---------------------------------------

export class PixelAgent {
  constructor(ort, session, meta, fileSize = 0) {
    this.ort = ort;
    this.session = session;
    this.meta = meta; // parsed pixel manifest block
    this.inputName = session.inputNames[0];
    this.fileSize = fileSize;
    this.params = 0; // filled by caller if wanted
  }

  // stack: Float32Array(4*96*96). Returns { action, logits, activations, aux }.
  async infer(stack) {
    const tensor = new this.ort.Tensor("float32", stack, [1, STACK, OBS_SIZE, OBS_SIZE]);
    const out = await this.session.run({ [this.inputName]: tensor });
    const get = (n) => out[n];
    const logits = Array.from(get("logits").data);
    const action = argmaxLogits(logits);
    const activations = {};
    for (const name of this.meta.activationOutputs) {
      const t = get(name);
      if (!t) continue;
      const dims = t.dims; // e.g. [1,16,23,23] or [1,256]
      activations[name] = {
        data: t.data,
        C: dims.length >= 4 ? dims[1] : 1,
        H: dims.length >= 4 ? dims[2] : 1,
        W: dims.length >= 4 ? dims[3] : dims[dims.length - 1],
        len: t.data.length,
      };
    }
    const aux = {
      rot: out.aux_rot ? argmaxLogits(Array.from(out.aux_rot.data)) : null,
      col: out.aux_col ? argmaxLogits(Array.from(out.aux_col.data)) : null,
    };
    return { action, logits, activations, aux };
  }
}

// ---- Real-time controller (loop-facing) -----------------------------------
//
// Exposes .deciding / .dead / .tick() so main.js's existing accumulator loop can
// drive it identically to the v1 Controller. Async inference is gated by
// `deciding` (the loop zeroes its accumulator while a decision is in flight),
// so play stays real-time without blocking rAF.

export class PixelController {
  constructor(env, renderEnv, agent, hooks = {}) {
    this.env = env;
    this.renderEnv = renderEnv;
    this.agent = agent;
    this.onPress = hooks.onPress || (() => {});
    this.onCommit = hooks.onCommit || (() => {});
    this.deciding = false;
    this.dead = false;
    this.pendingAction = null;
    this.history = [];
    this.lastObs = null;      // Uint8Array(96*96) the net last saw
    this.lastResult = null;   // { action, logits, activations, aux }
    this.lastDecisionMs = 0;
    this.decisions = 0;
  }

  async begin() {
    await this._decide();
  }

  // Render the current camera, roll the 4-history, run inference, stash the
  // chosen action. Sets `deciding` for the duration so the loop pauses.
  async _decide() {
    if (this.env.gameOver) { this.dead = true; return; }
    this.deciding = true;
    const obs = this.renderEnv(this.env);
    this.history = pushHistory(this.history, obs);
    this.lastObs = obs;
    const stack = stackToTensor(this.history);
    const t0 = performance.now();
    const res = await this.agent.infer(stack);
    this.lastDecisionMs = performance.now() - t0;
    this.lastResult = res;
    this.pendingAction = res.action;
    this.decisions += 1;
    this.deciding = false;
  }

  // One 30 Hz logic tick. On a decision tick with no action ready, kicks the
  // async decision and returns (loop zeroes its accumulator). Otherwise applies
  // the pending action (decision ticks only) and advances gravity.
  tick() {
    if (this.dead || this.deciding) return;
    if (this.env.gameOver) { this.dead = true; return; }
    if (this.env.isDecisionTick) {
      if (this.pendingAction === null) { this._decide(); return; }
      const a = this.pendingAction;
      this.env.apply_action(a);
      this.pendingAction = null;
      this.onPress(a);
    }
    const lock = this.env.tick();
    if (lock) this.onCommit(lock);
    if (this.env.gameOver) this.dead = true;
  }
}
