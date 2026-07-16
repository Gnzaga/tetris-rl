// Board rendering for the Pixel Agent tab (PLAN2.md §8).
//
// Draws the frame-layer board (frame_env.js state: `rows` row-ints, active piece
// pose, and `queue` preview) onto the big display canvas. This is the human-
// facing view; the network never sees it — it sees only render_obs.js's 96×96
// camera, shown separately in the model's-eye inset. The active piece is drawn
// in a distinct grey, echoing the 128 the camera renders.

import { WIDTH, HEIGHT } from "./features.js";

const LOCKED = "#7c5cff";     // violet locked stack
const ACTIVE = "#9aa0b4";     // grey active piece (mirrors obs value 128)
const GRID = "rgba(255,255,255,0.05)";
const BORDER = "rgba(124,92,255,0.45)";

export function renderPixelBoard(ctx, env, PIECES) {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  const cell = Math.min(Math.floor(W / WIDTH), Math.floor(H / HEIGHT));
  const ox = Math.floor((W - cell * WIDTH) / 2);
  const oy = Math.floor((H - cell * HEIGHT) / 2);

  ctx.fillStyle = "#0b0e15";
  ctx.fillRect(0, 0, W, H);

  // Faint grid.
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  for (let c = 0; c <= WIDTH; c++) {
    ctx.beginPath();
    ctx.moveTo(ox + c * cell + 0.5, oy);
    ctx.lineTo(ox + c * cell + 0.5, oy + HEIGHT * cell);
    ctx.stroke();
  }
  for (let r = 0; r <= HEIGHT; r++) {
    ctx.beginPath();
    ctx.moveTo(ox, oy + r * cell + 0.5);
    ctx.lineTo(ox + WIDTH * cell, oy + r * cell + 0.5);
    ctx.stroke();
  }

  // Locked stack.
  ctx.fillStyle = LOCKED;
  for (let r = 0; r < HEIGHT; r++) {
    const bits = env.rows[r];
    if (!bits) continue;
    for (let c = 0; c < WIDTH; c++) {
      if ((bits >> c) & 1) {
        ctx.fillRect(ox + c * cell + 1, oy + r * cell + 1, cell - 2, cell - 2);
      }
    }
  }

  // Active piece (only cells on the visible board).
  ctx.fillStyle = ACTIVE;
  const rot = PIECES[env.piece][env.rot];
  for (const [ro, co] of rot.cells) {
    const rr = env.row + ro;
    const cc = env.col + co;
    if (rr >= 0 && rr < HEIGHT && cc >= 0 && cc < WIDTH) {
      ctx.fillRect(ox + cc * cell + 1, oy + rr * cell + 1, cell - 2, cell - 2);
    }
  }

  // Border.
  ctx.strokeStyle = BORDER;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(ox - 1, oy - 1, WIDTH * cell + 2, HEIGHT * cell + 2);
}

export function renderPixelPreview(ctx, env, PIECES) {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);
  const next = env.queue[0];
  if (next == null) return;
  const rot = PIECES[next][0];
  const cell = 16;
  const ox = Math.floor((W - rot.width * cell) / 2);
  const oy = 8;
  ctx.fillStyle = LOCKED;
  for (const [ro, co] of rot.cells) {
    ctx.fillRect(ox + co * cell + 1, oy + ro * cell + 1, cell - 2, cell - 2);
  }
}
