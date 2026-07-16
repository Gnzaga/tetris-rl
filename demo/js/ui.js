// Rendering + stats UI for the demo (PLAN.md §10).
//
// Board/preview canvas rendering, the stats panel, and the training-curve mini
// chart. No game logic here — everything is driven by engine/controller state.

import { WIDTH, HEIGHT, CLEAR_POINTS } from "./engine.js";
import { HIDDEN_ROWS } from "./controller.js";
import { drawHeatmap } from "./heatmap.js";

export const CELL = 24;
export const BOARD_W = WIDTH * CELL; // 240
export const BOARD_H = (HEIGHT + HIDDEN_ROWS) * CELL; // 552

// Per-piece fill colors (I,O,T,S,Z,J,L).
const PIECE_COLORS = [
  "#3fd0e0", // I cyan
  "#f5d34a", // O yellow
  "#b46cf0", // T purple
  "#5fd35a", // S green
  "#f0584a", // Z red
  "#4a7cf0", // J blue
  "#f0a04a", // L orange
];
const GRID_BG = "#0d1017";
const HIDDEN_BG = "#141824";
const GRID_LINE = "#1c2230";
const LOCKED_EDGE = "rgba(255,255,255,0.15)";

function drawCell(ctx, row, col, color, edge = LOCKED_EDGE) {
  const x = col * CELL;
  const y = (row + HIDDEN_ROWS) * CELL;
  ctx.fillStyle = color;
  ctx.fillRect(x, y, CELL, CELL);
  ctx.strokeStyle = edge;
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, CELL - 1, CELL - 1);
}

export function renderBoard(ctx, engine, anim, decision, showHeatmap, pieces, dead) {
  // Backgrounds: hidden spawn strip then the play field.
  ctx.fillStyle = HIDDEN_BG;
  ctx.fillRect(0, 0, BOARD_W, HIDDEN_ROWS * CELL);
  ctx.fillStyle = GRID_BG;
  ctx.fillRect(0, HIDDEN_ROWS * CELL, BOARD_W, HEIGHT * CELL);

  // Grid lines over the play field only.
  ctx.strokeStyle = GRID_LINE;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let c = 0; c <= WIDTH; c++) {
    ctx.moveTo(c * CELL + 0.5, HIDDEN_ROWS * CELL);
    ctx.lineTo(c * CELL + 0.5, BOARD_H);
  }
  for (let r = 0; r <= HEIGHT; r++) {
    const y = (r + HIDDEN_ROWS) * CELL + 0.5;
    ctx.moveTo(0, y);
    ctx.lineTo(BOARD_W, y);
  }
  ctx.stroke();

  // Locked cells.
  const rows = engine.rows;
  for (let r = 0; r < HEIGHT; r++) {
    const v = rows[r];
    if (!v) continue;
    for (let c = 0; c < WIDTH; c++) {
      if (v & (1 << c)) drawCell(ctx, r, c, "#8a93a6");
    }
  }

  if (showHeatmap && decision) drawHeatmap(ctx, decision, engine, pieces, CELL);

  // Active animating piece.
  if (anim && anim.rotObj) {
    const color = PIECE_COLORS[anim.piece % 7];
    for (const [ro, co] of anim.rotObj.cells) {
      drawCell(ctx, anim.rowShown + ro, anim.colShown + co, color, "rgba(255,255,255,0.4)");
    }
  }

  if (dead) {
    ctx.fillStyle = "rgba(8,10,16,0.72)";
    ctx.fillRect(0, HIDDEN_ROWS * CELL, BOARD_W, HEIGHT * CELL);
    ctx.fillStyle = "#f0584a";
    ctx.font = "bold 22px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("GAME OVER", BOARD_W / 2, BOARD_H / 2);
    ctx.textAlign = "left";
  }
}

const PREVIEW_CELL = 16;
export function renderPreview(ctx, engine, pieces) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.fillStyle = GRID_BG;
  ctx.fillRect(0, 0, w, h);
  const queue = engine.previewPieces();
  const slot = h / 5;
  for (let i = 0; i < queue.length; i++) {
    const piece = queue[i];
    const rot = pieces[piece][0];
    const color = PIECE_COLORS[piece % 7];
    const pw = rot.width * PREVIEW_CELL;
    const ph = rot.height * PREVIEW_CELL;
    const ox = (w - pw) / 2;
    const oy = i * slot + (slot - ph) / 2;
    for (const [ro, co] of rot.cells) {
      ctx.fillStyle = color;
      ctx.fillRect(ox + co * PREVIEW_CELL, oy + ro * PREVIEW_CELL, PREVIEW_CELL, PREVIEW_CELL);
      ctx.strokeStyle = LOCKED_EDGE;
      ctx.strokeRect(ox + co * PREVIEW_CELL + 0.5, oy + ro * PREVIEW_CELL + 0.5, PREVIEW_CELL - 1, PREVIEW_CELL - 1);
    }
  }
}

// Rolling decision-latency + PPS tracker.
export class StatsTracker {
  constructor() {
    this.reset();
  }
  reset() {
    this.decisionMs = [];
    this.commitTimes = [];
    this.scoreTotal = 0;
  }
  onCommit(decisionMs, linesCleared = 0) {
    this.decisionMs.push(decisionMs);
    if (this.decisionMs.length > 30) this.decisionMs.shift();
    this.scoreTotal += CLEAR_POINTS[linesCleared] * 100;
    const now = performance.now();
    this.commitTimes.push(now);
    while (this.commitTimes.length && now - this.commitTimes[0] > 2000) this.commitTimes.shift();
  }
  avgMs() {
    if (!this.decisionMs.length) return 0;
    return this.decisionMs.reduce((a, b) => a + b, 0) / this.decisionMs.length;
  }
  pps() {
    if (this.commitTimes.length < 2) return 0;
    const span = (this.commitTimes[this.commitTimes.length - 1] - this.commitTimes[0]) / 1000;
    return span > 0 ? (this.commitTimes.length - 1) / span : 0;
  }
}

function fmtBytes(n) {
  if (!n) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

export function updateStats(els, engine, stats, runner) {
  els.lines.textContent = engine.lines.toLocaleString();
  els.pieces.textContent = engine.pieces.toLocaleString();
  els.score.textContent = stats.scoreTotal.toLocaleString();
  els.pps.textContent = stats.pps().toFixed(1);
  els.ms.textContent = stats.avgMs().toFixed(2);
  els.params.textContent = runner && runner.params ? runner.params.toLocaleString() : "—";
  els.size.textContent = runner ? fmtBytes(runner.fileSize) : "—";
}

// Training-curve mini line chart from manifest.curve.
export function drawCurve(ctx, curve) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0d1017";
  ctx.fillRect(0, 0, w, h);
  if (!curve || curve.length < 1) {
    ctx.fillStyle = "#5a6478";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText("no curve", 8, h / 2);
    return;
  }
  const pad = { l: 34, r: 8, t: 8, b: 18 };
  const xs = curve.map((p) => p.pieces_trained);
  const ys = curve.map((p) => p.eval_median_lines);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = 0;
  const ymax = Math.max(1, Math.max(...ys));
  const px = (x) => pad.l + ((x - xmin) / Math.max(1, xmax - xmin)) * (w - pad.l - pad.r);
  const py = (y) => h - pad.b - ((y - ymin) / (ymax - ymin)) * (h - pad.t - pad.b);

  // Axes.
  ctx.strokeStyle = "#2a3242";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();
  ctx.fillStyle = "#5a6478";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(String(Math.round(ymax)), 2, pad.t + 8);
  ctx.fillText("0", 2, h - pad.b);

  // Line + points.
  ctx.strokeStyle = "#3fd0e0";
  ctx.lineWidth = 2;
  ctx.beginPath();
  curve.forEach((p, i) => {
    const X = px(p.pieces_trained);
    const Y = py(p.eval_median_lines);
    if (i === 0) ctx.moveTo(X, Y);
    else ctx.lineTo(X, Y);
  });
  ctx.stroke();
  ctx.fillStyle = "#b46cf0";
  curve.forEach((p) => {
    ctx.beginPath();
    ctx.arc(px(p.pieces_trained), py(p.eval_median_lines), 2.5, 0, Math.PI * 2);
    ctx.fill();
  });
}
