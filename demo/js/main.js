// Demo entry point (PLAN.md §10 + PLAN2.md §8): wiring, the 30 Hz game loop,
// controls, self-tests, and tab switching. The v1 Play/Replay flow is unchanged;
// the Pixel Agent tab (shown only when the manifest carries a `pixel_agents`
// block) adds the real-time pixel-input / keypress agent with the model's-eye
// inset, MarI/O activation view, and keypress overlay.

import { makeEngine } from "./engine.js";
import { Controller } from "./controller.js";
import {
  createRunner,
  loadOrt,
  runSelfTest,
} from "./runner.js";
import {
  renderBoard,
  renderPreview,
  drawCurve,
  updateStats,
  StatsTracker,
} from "./ui.js";
import { ReplayTab } from "./replay.js";
import { makeFrameEnv } from "./frame_env.js";
import { makeRenderObs } from "./render_obs.js";
import {
  PixelAgent,
  PixelController,
  parsePixelManifest,
  stackToTensor,
  ACTION_TO_KEY,
  agentLogitBias,
} from "./pixel_agent.js";
import {
  drawFeatureMaps,
  drawFcStrip,
  drawWireGraph,
  drawObsInset,
} from "./activations.js";
import { Keypad } from "./keypad.js";
import { renderPixelBoard, renderPixelPreview } from "./pixel_ui.js";

const TICK = 1000 / 30; // 33.33 ms logic step
const SPEED_FACTORS = { "1": 1, "4": 4, "20": 20 };

const $ = (id) => document.getElementById(id);

const state = {
  paused: false,
  speed: "1", // "1" | "4" | "20" | "MAX"
  showHeatmap: false,
  activeTab: "play",
  pumping: false,
  acc: 0,
  lastTime: 0,
  showActivations: true,
};

let TetrisEngine, PIECES;
let manifest, ort;
let agentsById = {};
const runnerCache = new Map();
let playScene = null;
let replayTab = null;
const playStats = new StatsTracker();

// v2 pixel state.
let pixelMeta = null;      // parsed pixel_agents block (null => v1-only manifest)
let registryIndex = null;  // weights-free model registry (Model History panel)
let FrameEnv = null;
let renderEnv = null;
let pixelScene = null;
const pixelSessionCache = new Map();
let playKeypad = null;
let pixelKeypad = null;
let lastDrawnDecision = -1;

const els = {};

function cacheEls() {
  for (const id of [
    "board", "preview", "curve", "agent", "agentEval",
    "pause", "step", "restart", "seed", "heatmap",
    "statLines", "statScore", "statPieces", "statPps", "statMs", "statParams", "statSize",
    "selftest", "selftestPixel", "tabPlay", "tabReplay", "tabPixel",
    "panelPlay", "panelReplay", "panelPixel",
    "runSelect", "replaySelect", "live", "replayStatus", "replayBoard",
    "playKeypad",
    "pixelBoard", "pixelPreview", "pixelEye", "pixelAgent", "pixelEval",
    "pixelPause", "pixelRestart", "pixelSeed", "pixelActToggle", "pixelKeypad",
    "pixelCalibrated", "pixelCalNote", "pxPresses", "pixelHistory",
    "pxLines", "pxPieces", "pxDecisions", "pxAction", "pxMs", "pxAux",
    "pixelActivations", "actConv1", "actConv2", "actConv3", "actFc", "actWire",
  ]) {
    els[id] = $(id);
  }
  els.boardCtx = els.board.getContext("2d");
  els.previewCtx = els.preview.getContext("2d");
  els.curveCtx = els.curve.getContext("2d");
  els.replayCtx = els.replayBoard.getContext("2d");
  if (els.pixelHistory) els.pixelHistoryCtx = els.pixelHistory.getContext("2d");
  els.statEls = {
    lines: els.statLines, score: els.statScore, pieces: els.statPieces,
    pps: els.statPps, ms: els.statMs, params: els.statParams, size: els.statSize,
  };
}

function currentSeed() {
  const v = parseInt(els.seed.value, 10);
  return Number.isFinite(v) ? v >>> 0 : 1;
}

async function buildPlayScene() {
  const agentId = els.agent.value;
  const agent = agentsById[agentId];
  const seed = currentSeed();
  let runner = runnerCache.get(agentId);
  if (!runner || agent.type === "random") {
    runner = await createRunner(agent, ort, seed);
    if (agent.type !== "random") runnerCache.set(agentId, runner);
  }
  const engine = new TetrisEngine(seed);
  playStats.reset();
  const controller = new Controller(engine, runner, PIECES, (info, ms) => {
    playStats.onCommit(ms, info.linesCleared);
  });
  if (playKeypad) {
    playKeypad.reset();
    controller.onPress = (key) => playKeypad.press(key);
  }
  await controller.begin();
  playScene = { engine, controller, stats: playStats, runner, kind: "play" };
  updateAgentEval(agent);
}

function updateAgentEval(agent) {
  const e = agent.eval;
  if (!e || Object.keys(e).length === 0) {
    els.agentEval.textContent = agent.type === "random" ? "no eval stats" : "";
    return;
  }
  const bits = [];
  if (e.median_lines !== undefined) bits.push(`median ${Math.round(e.median_lines)}`);
  if (e.mean_lines !== undefined) bits.push(`mean ${e.mean_lines.toFixed(1)}`);
  if (e.pieces_per_game !== undefined) bits.push(`${Math.round(e.pieces_per_game)} pc/game`);
  els.agentEval.textContent = bits.join(" · ");
}

function activeScene() {
  if (state.activeTab === "play") return playScene;
  if (state.activeTab === "pixel") return pixelScene;
  return replayTab ? replayTab.scene : null;
}

// ---- Pixel agent scene -----------------------------------------------------

function pixelSeed() {
  const v = parseInt(els.pixelSeed.value, 10);
  return Number.isFinite(v) ? v >>> 0 : 1;
}

async function getPixelSession(agent) {
  let cached = pixelSessionCache.get(agent.id);
  if (cached) return cached;
  const path = "./models/" + agent.path;
  const session = await ort.InferenceSession.create(path, { executionProviders: ["wasm"] });
  const pa = new PixelAgent(ort, session, pixelMeta);
  pixelSessionCache.set(agent.id, pa);
  return pa;
}

async function buildPixelScene() {
  if (!pixelMeta || !ort) return;
  const agent = pixelMeta.byId[els.pixelAgent.value] || pixelMeta.byId[pixelMeta.final];
  const seed = pixelSeed();
  const pa = await getPixelSession(agent);
  const env = new FrameEnv(seed);
  if (pixelKeypad) pixelKeypad.reset();
  const stats = { presses: 0 };  // non-noop actions, for the live presses/piece stat
  const controller = new PixelController(env, renderEnv, pa, {
    logitBias: agentLogitBias(agent),
    calibrated: els.pixelCalibrated ? els.pixelCalibrated.checked : false,
    onPress: (action) => {
      if (action !== 0) stats.presses += 1;  // 0 == NOOP
      if (pixelKeypad) pixelKeypad.press(ACTION_TO_KEY[action]);
    },
  });
  await controller.begin();
  pixelScene = { env, controller, kind: "pixel", agent, stats };
  lastDrawnDecision = -1;
  updatePixelEval(agent);
}

function updatePixelEval(agent) {
  const e = agent.eval || {};
  const bits = [];
  if (e.median_lines !== undefined && e.median_lines !== null)
    bits.push(`median ${Math.round(e.median_lines)} lines`);
  if (e.pieces_per_game !== undefined && e.pieces_per_game !== null)
    bits.push(`${Math.round(e.pieces_per_game)} pc/game`);
  if (e.presses_per_piece !== undefined && e.presses_per_piece !== null)
    bits.push(`${Number(e.presses_per_piece).toFixed(1)} presses/piece raw`);
  if (e.thrash !== undefined && e.thrash !== null)
    bits.push(`thrash ${Number(e.thrash).toFixed(2)}`);
  els.pixelEval.textContent = bits.join(" · ") || "no eval stats";
}

// Model History panel: pieces/game (cyan, survival) and presses/piece (violet,
// spam level, capped) across the registry's pixel iterations, ordered by created
// date. Domain-mismatch is honest: gray-128 entries draw filled diamonds, the
// older camouflage-255 entries draw hollow circles. Reuses drawCurve's dark
// style. Median lines are ~0 everywhere so pieces/game is the informative axis.
const PRESS_CAP = 40;  // presses/piece axis cap (untrained ~97 is off-scale spam)
function drawPixelHistory(ctx, registry) {
  const w = ctx.canvas.width, h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0d1017";
  ctx.fillRect(0, 0, w, h);
  const entries = (registry && registry.entries || [])
    .filter((e) => e.family !== "v1-valuenet");
  if (entries.length < 1) {
    ctx.fillStyle = "#5a6478";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText("no history", 8, h / 2);
    return;
  }
  const pad = { l: 30, r: 30, t: 10, b: 16 };
  const n = entries.length;
  const px = (i) => pad.l + (n === 1 ? 0.5 : i / (n - 1)) * (w - pad.l - pad.r);
  const pcs = entries.map((e) => e.eval && e.eval.pieces_per_game);
  const pmax = Math.max(30, ...pcs.filter((v) => v != null));
  const pyPcs = (v) => h - pad.b - (v / pmax) * (h - pad.t - pad.b);
  const pyPress = (v) => h - pad.b - (Math.min(v, PRESS_CAP) / PRESS_CAP) * (h - pad.t - pad.b);

  // Axes.
  ctx.strokeStyle = "#2a3242"; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillStyle = "#3fd0e0"; ctx.fillText(String(Math.round(pmax)), 2, pad.t + 7);
  ctx.fillStyle = "#b46cf0"; ctx.fillText(String(PRESS_CAP), w - pad.r + 3, pad.t + 7);

  const line = (accessor, py, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    entries.forEach((e, i) => {
      const v = accessor(e);
      if (v == null) return;
      const X = px(i), Y = py(v);
      if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
    });
    ctx.stroke();
    entries.forEach((e, i) => {
      const v = accessor(e);
      if (v == null) return;
      const X = px(i), Y = py(v);
      ctx.fillStyle = color; ctx.strokeStyle = color; ctx.lineWidth = 1.4;
      ctx.beginPath();
      if (e.domain === "gray-128") {           // filled diamond
        ctx.moveTo(X, Y - 3); ctx.lineTo(X + 3, Y); ctx.lineTo(X, Y + 3);
        ctx.lineTo(X - 3, Y); ctx.closePath(); ctx.fill();
      } else {                                   // hollow circle (camouflage-255)
        ctx.arc(X, Y, 2.6, 0, Math.PI * 2); ctx.stroke();
      }
    });
  };
  line((e) => e.eval && e.eval.pieces_per_game, pyPcs, "#3fd0e0");
  line((e) => e.eval && e.eval.presses_per_piece, pyPress, "#b46cf0");
}

// ---- Game loop -------------------------------------------------------------

async function pumpMax(scene) {
  state.pumping = true;
  const budget = performance.now() + 10;
  while (
    scene === activeScene() && state.speed === "MAX" && !state.paused &&
    scene.controller && !scene.controller.dead && performance.now() < budget
  ) {
    const ok = await scene.controller.stepMax();
    if (!ok) break;
  }
  state.pumping = false;
}

function frame(now) {
  requestAnimationFrame(frame);
  const dt = now - state.lastTime;
  state.lastTime = now;
  const scene = activeScene();
  if (scene && scene.controller && !state.paused) {
    const c = scene.controller;
    // MAX is a v1-only Play/Replay affordance (pixel is real-time by definition).
    if (state.speed === "MAX" && scene.kind !== "pixel") {
      if (!state.pumping) pumpMax(scene);
    } else {
      const factor = SPEED_FACTORS[state.speed] || 1;
      state.acc += Math.min(dt, 200) * factor;
      let guard = 0;
      while (state.acc >= TICK && !c.deciding && !c.dead && guard < 800) {
        state.acc -= TICK;
        c.tick();
        guard++;
      }
      if (c.deciding) state.acc = 0;
    }
  }
  render();
}

function render() {
  if (state.activeTab === "play") {
    if (playScene) {
      const c = playScene.controller;
      renderBoard(els.boardCtx, playScene.engine, c.anim, c.decision, state.showHeatmap, PIECES, c.dead);
      renderPreview(els.previewCtx, playScene.engine, PIECES);
      updateStats(els.statEls, playScene.engine, playStats, playScene.runner);
    }
  } else if (state.activeTab === "pixel") {
    renderPixel();
  } else if (replayTab && replayTab.scene) {
    const s = replayTab.scene;
    renderBoard(els.replayCtx, s.engine, s.controller.anim, null, false, PIECES, s.controller.dead);
  }
}

function renderPixel() {
  if (!pixelScene) return;
  const c = pixelScene.controller;
  renderPixelBoard(els.pixelBoard.getContext("2d"), pixelScene.env, PIECES);
  renderPixelPreview(els.pixelPreview.getContext("2d"), pixelScene.env, PIECES);
  if (c.lastObs) drawObsInset(els.pixelEye.getContext("2d"), els.pixelEye, c.lastObs);

  els.pxLines.textContent = pixelScene.env.lines;
  els.pxPieces.textContent = pixelScene.env.pieces;
  els.pxDecisions.textContent = c.decisions;
  els.pxMs.textContent = c.lastDecisionMs.toFixed(2);
  if (els.pxPresses) {
    const pcs = pixelScene.env.pieces;
    const pp = pcs > 0 ? pixelScene.stats.presses / pcs : 0;
    els.pxPresses.textContent = pp.toFixed(2);
  }
  if (c.lastResult) {
    els.pxAction.textContent = pixelMeta.legend[c.lastResult.action] ?? "—";
    const aux = c.lastResult.aux;
    els.pxAux.textContent = (aux && aux.rot != null)
      ? `rot ${aux.rot} · col ${aux.col}` : "—";
  }

  // Activation view — redraw only on a fresh inference (≈10 Hz).
  if (state.showActivations && c.lastResult && c.decisions !== lastDrawnDecision) {
    lastDrawnDecision = c.decisions;
    const acts = c.lastResult.activations;
    if (acts.conv1) drawFeatureMaps(els.actConv1.getContext("2d"), els.actConv1, acts.conv1, 4);
    if (acts.conv2) drawFeatureMaps(els.actConv2.getContext("2d"), els.actConv2, acts.conv2, 8);
    if (acts.conv3) drawFeatureMaps(els.actConv3.getContext("2d"), els.actConv3, acts.conv3, 8);
    if (acts.fc) drawFcStrip(els.actFc.getContext("2d"), els.actFc, acts.fc.data, acts.fc.len);
    if (acts.fc && pixelMeta.fcWeight) {
      drawWireGraph(els.actWire.getContext("2d"), els.actWire, acts.fc.data,
        pixelMeta.fcWeight, c.lastResult.logits, c.lastResult.action, pixelMeta.legend);
    }
  }
}

// ---- Controls --------------------------------------------------------------

function setSpeed(sp) {
  state.speed = sp;
  state.acc = 0;
  for (const btn of document.querySelectorAll(".speed-btn")) {
    btn.classList.toggle("active", btn.dataset.speed === sp);
  }
  for (const btn of document.querySelectorAll(".pixel-speed-btn")) {
    btn.classList.toggle("active", btn.dataset.speed === sp);
  }
}

function wireControls() {
  els.agent.addEventListener("change", async () => {
    els.agent.disabled = true;
    await buildPlayScene();
    els.agent.disabled = false;
  });
  for (const btn of document.querySelectorAll(".speed-btn")) {
    btn.addEventListener("click", () => setSpeed(btn.dataset.speed));
  }
  els.pause.addEventListener("click", () => {
    state.paused = !state.paused;
    els.pause.textContent = state.paused ? "Resume" : "Pause";
  });
  els.step.addEventListener("click", () => {
    state.paused = true;
    els.pause.textContent = "Resume";
    const s = activeScene();
    if (s && s.controller && !s.controller.dead && !s.controller.deciding && s.controller.anim) {
      s.controller.forceCommit();
    }
    render();
  });
  els.restart.addEventListener("click", async () => {
    await buildPlayScene();
    state.paused = false;
    els.pause.textContent = "Pause";
  });
  els.seed.addEventListener("change", async () => {
    await buildPlayScene();
  });
  els.heatmap.addEventListener("change", () => {
    state.showHeatmap = els.heatmap.checked;
  });
  els.tabPlay.addEventListener("click", () => switchTab("play"));
  els.tabReplay.addEventListener("click", () => switchTab("replay"));
  els.tabPixel.addEventListener("click", () => switchTab("pixel"));

  // Pixel controls.
  if (pixelMeta) {
    els.pixelAgent.addEventListener("change", async () => {
      els.pixelAgent.disabled = true;
      await buildPixelScene();
      els.pixelAgent.disabled = false;
    });
    for (const btn of document.querySelectorAll(".pixel-speed-btn")) {
      btn.addEventListener("click", () => setSpeed(btn.dataset.speed));
    }
    els.pixelPause.addEventListener("click", () => {
      state.paused = !state.paused;
      els.pixelPause.textContent = state.paused ? "Resume" : "Pause";
    });
    els.pixelRestart.addEventListener("click", async () => {
      await buildPixelScene();
      state.paused = false;
      els.pixelPause.textContent = "Pause";
    });
    els.pixelSeed.addEventListener("change", async () => { await buildPixelScene(); });
    els.pixelActToggle.addEventListener("change", () => {
      state.showActivations = els.pixelActToggle.checked;
      els.pixelActivations.classList.toggle("hidden", !state.showActivations);
    });
    if (els.pixelCalibrated) {
      els.pixelCalibrated.addEventListener("change", () => {
        // Applies live to the running controller (no rebuild needed).
        if (pixelScene && pixelScene.controller) {
          pixelScene.controller.calibrated = els.pixelCalibrated.checked;
        }
        if (els.pixelCalNote) {
          els.pixelCalNote.textContent = els.pixelCalibrated.checked
            ? "reduces button-spam" : "raw net — watch the spam";
        }
      });
    }
  }
}

function switchTab(tab) {
  if (tab === "pixel" && !pixelMeta) return;
  state.activeTab = tab;
  state.paused = false;
  els.tabPlay.classList.toggle("active", tab === "play");
  els.tabReplay.classList.toggle("active", tab === "replay");
  els.tabPixel.classList.toggle("active", tab === "pixel");
  els.panelPlay.classList.toggle("hidden", tab !== "play");
  els.panelReplay.classList.toggle("hidden", tab !== "replay");
  els.panelPixel.classList.toggle("hidden", tab !== "pixel");
  // Pixel is 1×/4× only; clamp a v1 MAX/20× selection on entry.
  if (tab === "pixel" && !(state.speed === "1" || state.speed === "4")) setSpeed("1");
  els.pause.textContent = "Pause";
  els.pixelPause.textContent = "Pause";
}

// ---- Self-tests ------------------------------------------------------------

async function doSelfTest() {
  const onnxAgents = manifest.agents.filter((a) => a.type === "onnx");
  if (!onnxAgents.length || !manifest.selftest) {
    els.selftest.textContent = "self-test: skipped (no ONNX model)";
    els.selftest.className = "selftest warn";
    return;
  }
  if (!ort) {
    els.selftest.textContent = "self-test: ORT unavailable";
    els.selftest.className = "selftest fail";
    return;
  }
  const finalAgent = onnxAgents[onnxAgents.length - 1];
  try {
    let runner = runnerCache.get(finalAgent.id);
    if (!runner) {
      runner = await createRunner(finalAgent, ort, 1);
      runnerCache.set(finalAgent.id, runner);
    }
    const res = await runSelfTest(ort, runner.session, manifest.selftest);
    if (res.pass) {
      els.selftest.textContent = `self-test: PASS (max err ${res.maxErr.toExponential(1)})`;
      els.selftest.className = "selftest pass";
    } else {
      els.selftest.textContent = `self-test: FAIL (max err ${res.maxErr.toExponential(1)})`;
      els.selftest.className = "selftest fail";
    }
  } catch (e) {
    els.selftest.textContent = `self-test: ERROR ${e.message}`;
    els.selftest.className = "selftest fail";
  }
}

// 2 fixed obs stacks -> final pixel ONNX; compare logits within tol (1e-3).
async function doPixelSelfTest() {
  if (!pixelMeta || !pixelMeta.selftest || !ort) return;
  els.selftestPixel.classList.remove("hidden");
  try {
    const sidecar = await fetch("./models/" + pixelMeta.selftest.path).then((r) => r.json());
    const raw = Uint8Array.from(atob(sidecar.stacks_b64), (ch) => ch.charCodeAt(0));
    const [n] = sidecar.shape;
    const frameLen = 4 * 96 * 96;
    const finalAgent = pixelMeta.byId[pixelMeta.final];
    const pa = await getPixelSession(finalAgent);
    const tol = sidecar.tol ?? 1e-3;
    let maxErr = 0;
    for (let s = 0; s < n; s++) {
      // Reconstruct the [4,96,96] stack (already channel-major) and normalize.
      const stackU8 = raw.subarray(s * frameLen, (s + 1) * frameLen);
      const tensor = new Float32Array(frameLen);
      for (let i = 0; i < frameLen; i++) tensor[i] = stackU8[i] / 255;
      const out = await pa.session.run({ [pa.inputName]: new ort.Tensor("float32", tensor, [1, 4, 96, 96]) });
      const got = out.logits.data;
      const exp = sidecar.expected_logits[s];
      for (let i = 0; i < exp.length; i++) maxErr = Math.max(maxErr, Math.abs(got[i] - exp[i]));
    }
    const pass = maxErr <= tol;
    els.selftestPixel.textContent = `pixel self-test: ${pass ? "PASS" : "FAIL"} (max err ${maxErr.toExponential(1)})`;
    els.selftestPixel.className = `selftest ${pass ? "pass" : "fail"}`;
  } catch (e) {
    els.selftestPixel.textContent = `pixel self-test: ERROR ${e.message}`;
    els.selftestPixel.className = "selftest fail";
  }
}

// ---- Init ------------------------------------------------------------------

async function init() {
  cacheEls();
  const piecesJson = await fetch("../shared/pieces.json").then((r) => r.json());
  const built = makeEngine(piecesJson);
  TetrisEngine = built.TetrisEngine;
  PIECES = built.PIECES;

  manifest = await fetch("./models/manifest.json").then((r) => r.json());
  for (const a of manifest.agents) agentsById[a.id] = a;
  pixelMeta = parsePixelManifest(manifest);
  registryIndex = await fetch("./models/registry.json")
    .then((r) => r.json()).catch(() => null);  // optional; Model History panel

  els.selftest.textContent = "loading ONNX runtime…";
  try {
    ort = await loadOrt();
  } catch (e) {
    ort = null;
    els.selftest.textContent = `ORT load failed: ${e.message}`;
    els.selftest.className = "selftest fail";
  }

  // Agent dropdown.
  els.agent.innerHTML = "";
  for (const a of manifest.agents) {
    const opt = document.createElement("option");
    opt.value = a.id;
    opt.textContent = a.label;
    if (a.type === "onnx" && !ort) opt.disabled = true;
    els.agent.appendChild(opt);
  }
  els.agent.value = agentsById["cem"] ? "cem" : manifest.agents[0].id;

  drawCurve(els.curveCtx, manifest.curve);

  // Keypads.
  playKeypad = new Keypad(els.playKeypad);

  // Pixel tab setup (only when the manifest carries pixel agents).
  if (pixelMeta) {
    FrameEnv = makeFrameEnv(piecesJson).FrameEnv;
    renderEnv = makeRenderObs(piecesJson).renderEnv;
    pixelKeypad = new Keypad(els.pixelKeypad);
    els.tabPixel.classList.remove("hidden");
    els.pixelAgent.innerHTML = "";
    for (const a of pixelMeta.agents) {
      const opt = document.createElement("option");
      opt.value = a.id;
      opt.textContent = a.label;
      if (!ort) opt.disabled = true;
      els.pixelAgent.appendChild(opt);
    }
    els.pixelAgent.value = pixelMeta.final;
    // Calibration ("calibrated presses") starts ON for the chosen default.
    if (els.pixelCalibrated) els.pixelCalibrated.checked = !!pixelMeta.calibratedDefault;
    if (els.pixelHistoryCtx) drawPixelHistory(els.pixelHistoryCtx, registryIndex);
  }

  wireControls();
  setSpeed("1");

  await buildPlayScene();

  replayTab = new ReplayTab({
    makeEngine: TetrisEngine,
    pieces: PIECES,
    els: {
      runSelect: els.runSelect,
      replaySelect: els.replaySelect,
      liveToggle: els.live,
      status: els.replayStatus,
    },
    onScene: () => {},
  });
  replayTab.init();

  if (pixelMeta && ort) {
    try { await buildPixelScene(); } catch (e) { /* surfaced via pixel self-test */ }
  }

  doSelfTest();
  doPixelSelfTest();

  state.lastTime = performance.now();
  requestAnimationFrame(frame);
}

init();
