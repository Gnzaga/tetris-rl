// Replay tab (PLAN.md §10).
//
// Reads runs/index.json (written by the trainers), lists each run's
// replays/index.json, and animates any eval replay through the SAME engine +
// controller as live play by feeding the recorded move list via ReplayRunner.
// A "live" toggle re-polls the selected run's replay index every 10 s.

import { Controller } from "./controller.js";
import { ReplayRunner } from "./runner.js";
import { StatsTracker } from "./ui.js";

const RUNS_INDEX = "../runs/index.json";
const POLL_MS = 10000;

export class ReplayTab {
  constructor({ makeEngine, pieces, els, onScene }) {
    this.makeEngine = makeEngine;
    this.pieces = pieces;
    this.els = els; // { runSelect, replaySelect, liveToggle, status }
    this.onScene = onScene;
    this.stats = new StatsTracker();
    this.scene = null;
    this.currentRun = null;
    this.pollTimer = null;

    els.runSelect.addEventListener("change", () => this.selectRun(els.runSelect.value));
    els.replaySelect.addEventListener("change", () => this.loadReplay(els.replaySelect.value));
    els.liveToggle.addEventListener("change", () => this.updatePolling());
  }

  async init() {
    try {
      const runs = await fetch(RUNS_INDEX).then((r) => r.json());
      this.els.runSelect.innerHTML = "";
      for (const run of runs) {
        const opt = document.createElement("option");
        opt.value = run.name;
        opt.textContent = `${run.name} (${run.phase}, ${run.num_replays} replays)`;
        this.els.runSelect.appendChild(opt);
      }
      if (runs.length) await this.selectRun(runs[0].name);
    } catch (e) {
      this.els.status.textContent = `runs/index.json unavailable: ${e.message}`;
    }
  }

  async selectRun(name) {
    this.currentRun = name;
    await this.refreshReplayList(true);
    this.updatePolling();
  }

  async refreshReplayList(autoLoadFirst = false) {
    if (!this.currentRun) return;
    try {
      const list = await fetch(`../runs/${this.currentRun}/replays/index.json`).then((r) => r.json());
      const prevValue = this.els.replaySelect.value;
      this.els.replaySelect.innerHTML = "";
      for (const rep of list) {
        const opt = document.createElement("option");
        opt.value = rep.file;
        opt.textContent = `${rep.pieces_trained} pieces — median ${rep.median_lines}`;
        this.els.replaySelect.appendChild(opt);
      }
      if (autoLoadFirst && list.length) {
        this.els.replaySelect.value = list[list.length - 1].file;
        await this.loadReplay(this.els.replaySelect.value);
      } else if (prevValue && [...this.els.replaySelect.options].some((o) => o.value === prevValue)) {
        this.els.replaySelect.value = prevValue;
      }
    } catch (e) {
      this.els.status.textContent = `replay index unavailable: ${e.message}`;
    }
  }

  async loadReplay(file) {
    if (!file || !this.currentRun) return;
    try {
      const data = await fetch(`../runs/${this.currentRun}/replays/${file}`).then((r) => r.json());
      const engine = new this.makeEngine(data.seed);
      this.stats.reset();
      const runner = new ReplayRunner(data.moves);
      const controller = new Controller(engine, runner, this.pieces, (info, ms) => {
        this.stats.onCommit(ms, info.linesCleared);
      });
      await controller.begin();
      this.scene = { engine, controller, stats: this.stats, runner: null, kind: "replay", meta: data };
      this.els.status.textContent = `seed ${data.seed} — final ${data.final.lines} lines / ${data.final.pieces} pieces`;
      this.onScene(this.scene);
    } catch (e) {
      this.els.status.textContent = `replay load failed: ${e.message}`;
    }
  }

  updatePolling() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    if (this.els.liveToggle.checked && this.currentRun) {
      this.pollTimer = setInterval(() => this.refreshReplayList(false), POLL_MS);
    }
  }

  stopPolling() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }
}
