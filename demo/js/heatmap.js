// Top-5 candidate-placement heatmap overlay (PLAN.md §10).
//
// Draws ghost outlines of the five highest-scoring legal placements for the
// current piece, tinted by softmax-normalized score, with the raw score/value
// printed beside each. Purely a render overlay — reads the decision produced by
// the runner, computes resting positions with the same drop logic as the engine.

import { softmax, topKIndices } from "./runner.js";
import { dropRowOf, HIDDEN_ROWS } from "./controller.js";

// Interpolate cyan (low) -> magenta (high) by softmax weight t in [0,1].
function tint(t) {
  const r = Math.round(40 + t * 215);
  const g = Math.round(200 - t * 140);
  const b = Math.round(230 - t * 30);
  return [r, g, b];
}

export function drawHeatmap(ctx, decision, engine, pieces, cell) {
  if (!decision || !decision.placements || decision.placements.length === 0) return;
  const { placements, scores } = decision;
  const k = Math.min(5, placements.length);
  const top = topKIndices(scores, k);
  const topScores = top.map((i) => scores[i]);
  const weights = softmax(topScores);
  const piece = engine.current;
  const rots = pieces[piece];

  ctx.save();
  ctx.font = "11px ui-monospace, monospace";
  ctx.textBaseline = "middle";

  for (let n = k - 1; n >= 0; n--) {
    const idx = top[n];
    const [rot, col] = placements[idx];
    const rotObj = rots[rot];
    const topRow = dropRowOf(engine.rows, rotObj, col);
    if (topRow < 0) continue;
    const [r, g, b] = tint(weights[n]);
    const alpha = 0.25 + 0.6 * weights[n];
    ctx.lineWidth = n === 0 ? 2.5 : 1.5;
    ctx.strokeStyle = `rgba(${r},${g},${b},${Math.min(1, alpha + 0.25)})`;
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha * 0.35})`;

    let minX = Infinity;
    let minY = Infinity;
    for (const [ro, co] of rotObj.cells) {
      const x = (col + co) * cell;
      const y = (topRow + ro + HIDDEN_ROWS) * cell;
      ctx.fillRect(x, y, cell, cell);
      ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      if (x < minX) minX = x;
      if (y < minY) minY = y;
    }
    // Value label beside the top-left cell of this ghost.
    const label = scores[idx].toFixed(1);
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    const tw = ctx.measureText(label).width + 6;
    const lx = Math.min(minX, cell * 10 - tw);
    const ly = Math.max(minY - 8, 8);
    ctx.fillRect(lx, ly - 7, tw, 14);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillText(label, lx + 3, ly);
  }
  ctx.restore();
}
