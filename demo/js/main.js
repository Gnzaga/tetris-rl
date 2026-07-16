// Demo entry point (PLAN.md §10): wiring, the 30 Hz game loop, controls,
// self-test, and tab switching.

import { makeEngine } from "./engine.js";
import { Controller } from "./controller.js";
import {
  createRunner,
  loadOrt,
  runSelfTest,
  RandomRunner,
} from "./runner.js";
import {
  renderBoard,
  renderPreview,
  drawCurve,
  updateStats,
  StatsTracker,
} from "./ui.js";
import { ReplayTab } from "./replay.js";

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
};

let TetrisEngine, PIECES;
let manifest, ort;
let agentsById = {};
const runnerCache = new Map();
let playScene = null;
let replayTab = null;
const playStats = new StatsTracker();

const els = {};

function cacheEls() {
  for (const id of [
    "board", "preview", "curve", "agent", "agentEval",
    "pause", "step", "restart", "seed", "heatmap",
    "statLines", "statScore", "statPieces", "statPps", "statMs", "statParams", "statSize",
    "selftest", "tabPlay", "tabReplay", "panelPlay", "panelReplay",
    "runSelect", "replaySelect", "live", "replayStatus", "replayBoard",
  ]) {
    els[id] = $(id);
  }
  els.boardCtx = els.board.getContext("2d");
  els.previewCtx = els.preview.getContext("2d");
  els.curveCtx = els.curve.getContext("2d");
  els.replayCtx = els.replayBoard.getContext("2d");
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
  return state.activeTab === "play" ? playScene : (replayTab ? replayTab.scene : null);
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
    if (state.speed === "MAX") {
      if (!state.pumping) pumpMax(scene);
    } else {
      state.acc += Math.min(dt, 200) * SPEED_FACTORS[state.speed];
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
  } else if (replayTab && replayTab.scene) {
    const s = replayTab.scene;
    renderBoard(els.replayCtx, s.engine, s.controller.anim, null, false, PIECES, s.controller.dead);
  }
}

// ---- Controls --------------------------------------------------------------

function setSpeed(sp) {
  state.speed = sp;
  state.acc = 0;
  for (const btn of document.querySelectorAll(".speed-btn")) {
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
}

function switchTab(tab) {
  state.activeTab = tab;
  els.tabPlay.classList.toggle("active", tab === "play");
  els.tabReplay.classList.toggle("active", tab === "replay");
  els.panelPlay.classList.toggle("hidden", tab !== "play");
  els.panelReplay.classList.toggle("hidden", tab !== "replay");
}

// ---- Self-test -------------------------------------------------------------

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

// ---- Init ------------------------------------------------------------------

async function init() {
  cacheEls();
  const piecesJson = await fetch("../shared/pieces.json").then((r) => r.json());
  const built = makeEngine(piecesJson);
  TetrisEngine = built.TetrisEngine;
  PIECES = built.PIECES;

  manifest = await fetch("./models/manifest.json").then((r) => r.json());
  for (const a of manifest.agents) agentsById[a.id] = a;

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
  // Default to a strong classical agent for an immediate, non-toppling display.
  els.agent.value = agentsById["cem"] ? "cem" : manifest.agents[0].id;

  drawCurve(els.curveCtx, manifest.curve);
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

  doSelfTest();

  state.lastTime = performance.now();
  requestAnimationFrame(frame);
}

init();
