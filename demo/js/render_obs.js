// 96x96 grayscale observation rasterizer (PLAN2.md §1, Phase C).
//
// Bit-exact port of tetris/render_obs.py. Renders the "camera" the pixel agent
// sees into a plain Uint8Array(96*96) — no DOM / canvas required (Phase F can
// blit it to a canvas with putImageData). Filled stack cells, the active piece,
// and the next-piece preview all render 255 (the active piece is NOT visually
// distinct from the stack); empty is 0; the board border is 255.
//
// Frozen pixel layout (identical to render_obs.py; changing any constant breaks
// the bit-exact CRC32 fixtures):
//   * Canvas 96x96, row-major: buf[y * 96 + x] (y down, x right).
//   * Board 10x20 cells at 4 px/cell (40x80 px), interior top-left (8, 8), so
//     interior x in [8,47], y in [8,87]. Cell (r, c) -> 4x4 rect top-left
//     (8 + 4c, 8 + 4r).
//   * Border: 1 px white ring around the board = perimeter of the rectangle
//     with inclusive corners (7, 7) and (48, 88).
//   * Preview: NEXT piece (queue[0]) at its rotation-0 bbox, top-left aligned in
//     a 20x20 region with top-left (56, 8), 4 px/cell, no border. Preview cell
//     (r, c) -> 4x4 rect top-left (56 + 4c, 8 + 4r).
// Active-piece cells at negative board rows are simply not drawn.

import { makeEngine } from "./engine.js";
import { WIDTH, HEIGHT } from "./features.js";

export const OBS_SIZE = 96;
export const CELL = 4;

export const BOARD_X = 8;
export const BOARD_Y = 8;

export const BORDER_X0 = BOARD_X - 1; // 7
export const BORDER_Y0 = BOARD_Y - 1; // 7
export const BORDER_X1 = BOARD_X + WIDTH * CELL; // 48
export const BORDER_Y1 = BOARD_Y + HEIGHT * CELL; // 88

export const PREVIEW_X = 56;
export const PREVIEW_Y = 8;

const FILLED = 255;

export function makeRenderObs(piecesJson) {
  const { PIECES } = makeEngine(piecesJson);

  function fillCell(buf, x0, y0) {
    for (let dy = 0; dy < CELL; dy++) {
      const base = (y0 + dy) * OBS_SIZE + x0;
      for (let dx = 0; dx < CELL; dx++) buf[base + dx] = FILLED;
    }
  }

  // renderObs(rows, piece, rot, col, row, nextPiece) -> Uint8Array(96*96).
  function renderObs(rows, piece, rot, col, row, nextPiece) {
    const buf = new Uint8Array(OBS_SIZE * OBS_SIZE);

    // Border ring (perimeter of (7,7)..(48,88)).
    for (let x = BORDER_X0; x <= BORDER_X1; x++) {
      buf[BORDER_Y0 * OBS_SIZE + x] = FILLED;
      buf[BORDER_Y1 * OBS_SIZE + x] = FILLED;
    }
    for (let y = BORDER_Y0; y <= BORDER_Y1; y++) {
      buf[y * OBS_SIZE + BORDER_X0] = FILLED;
      buf[y * OBS_SIZE + BORDER_X1] = FILLED;
    }

    // Locked stack cells.
    for (let r = 0; r < HEIGHT; r++) {
      const bits = rows[r];
      if (!bits) continue;
      const y0 = BOARD_Y + r * CELL;
      for (let c = 0; c < WIDTH; c++) {
        if ((bits >> c) & 1) fillCell(buf, BOARD_X + c * CELL, y0);
      }
    }

    // Active piece (cells above the board are not drawn).
    for (const [ro, co] of PIECES[piece][rot].cells) {
      const rr = row + ro;
      const cc = col + co;
      if (rr >= 0 && rr < HEIGHT && cc >= 0 && cc < WIDTH) {
        fillCell(buf, BOARD_X + cc * CELL, BOARD_Y + rr * CELL);
      }
    }

    // Next-piece preview at its rotation-0 bbox, top-left aligned, no border.
    for (const [ro, co] of PIECES[nextPiece][0].cells) {
      fillCell(buf, PREVIEW_X + co * CELL, PREVIEW_Y + ro * CELL);
    }

    return buf;
  }

  // renderEnv(env) -> Uint8Array: preview shows env.queue[0] (the NEXT piece).
  function renderEnv(env) {
    return renderObs(env.rows, env.piece, env.rot, env.col, env.row, env.queue[0]);
  }

  return { renderObs, renderEnv, PIECES };
}
