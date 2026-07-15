// Bitboard Tetris simulator (PLAN.md §2).
//
// Board: 10 columns x 20 rows, row 0 top / row 19 bottom, each row a 10-bit int
// (bit c set = column c filled). The engine's only transition is the atomic
// `step(rotation, column)`; a placement drops straight to its resting position.
//
// Piece rotation tables live in shared/pieces.json — never duplicated in code.
// `makeEngine(piecesJson)` binds a parsed JSON object (fetched in the browser,
// fs-read in Node) and returns the engine class plus the built rotation tables.
// This is a bit-exact port of tetris/engine.py; transitions and the full
// 8-feature vector match the Python engine exactly.

import { boardFeatures, WIDTH, HEIGHT, FULL_ROW } from "./features.js";
import { Mulberry32, SevenBag } from "./rng.js";

export { WIDTH, HEIGHT, FULL_ROW };
export const CLEAR_POINTS = [0, 1, 3, 5, 8];

function buildRotation(rot) {
  const width = rot.width;
  const height = rot.height;
  const cells = rot.cells.map(([r, c]) => [r, c]);
  // bottom[pc] = lowest (max) row offset among cells in piece-column pc.
  const bottom = new Array(width).fill(0);
  for (let pc = 0; pc < width; pc++) {
    let mx = 0;
    for (const [r, c] of cells) if (c === pc && r > mx) mx = r;
    bottom[pc] = mx;
  }
  return { width, height, cells, bottom, nPlacements: WIDTH - width + 1 };
}

export function makeEngine(piecesJson) {
  const PIECES = piecesJson.pieces.map((p) => p.rotations.map(buildRotation));

  class TetrisEngine {
    constructor(seed, preview = 5) {
      this.rng = new Mulberry32(seed);
      this.bag = new SevenBag(this.rng);
      this.preview = preview;
      this.rows = new Array(HEIGHT).fill(0);
      this.queue = [];
      this._fillQueue();
      this.current = this.queue.shift();
      this._fillQueue();
      this.lines = 0;
      this.pieces = 0;
      this.gameOver = false;
    }

    _fillQueue() {
      while (this.queue.length < this.preview) this.queue.push(this.bag.nextPiece());
    }

    previewPieces() {
      return this.queue.slice(0, this.preview);
    }

    _colTop() {
      // Topmost filled row per column; HEIGHT (=20) if the column is empty.
      const top = new Array(WIDTH).fill(HEIGHT);
      let remaining = WIDTH;
      for (let r = 0; r < HEIGHT && remaining > 0; r++) {
        const row = this.rows[r];
        if (!row) continue;
        for (let c = 0; c < WIDTH; c++) {
          if (top[c] === HEIGHT && row & (1 << c)) {
            top[c] = r;
            remaining--;
          }
        }
      }
      return top;
    }

    _dropRow(rot, col, colTop) {
      // Resting top-row of the bounding box (may be negative => illegal).
      let t = HEIGHT;
      for (let pc = 0; pc < rot.width; pc++) {
        const v = colTop[col + pc] - 1 - rot.bottom[pc];
        if (v < t) t = v;
      }
      return t;
    }

    legalPlacements(piece) {
      if (piece === undefined || piece === null) piece = this.current;
      const colTop = this._colTop();
      const out = [];
      const rots = PIECES[piece];
      for (let rotIdx = 0; rotIdx < rots.length; rotIdx++) {
        const rot = rots[rotIdx];
        for (let col = 0; col < rot.nPlacements; col++) {
          if (this._dropRow(rot, col, colTop) >= 0) out.push([rotIdx, col]);
        }
      }
      return out;
    }

    _apply(rot, col, topRow) {
      // Lock the piece at topRow and clear lines. Does not mutate engine state.
      const pre = this.rows.slice();
      for (const [ro, co] of rot.cells) pre[topRow + ro] |= 1 << (col + co);

      const fullRows = [];
      for (let r = topRow; r < topRow + rot.height; r++) {
        if (pre[r] === FULL_ROW) fullRows.push(r);
      }
      const lines = fullRows.length;

      let eroded = 0;
      if (lines) {
        let pieceCellsCleared = 0;
        for (const [ro] of rot.cells) {
          if (fullRows.includes(topRow + ro)) pieceCellsCleared++;
        }
        eroded = lines * pieceCellsCleared;
      }

      // landing height: mean of rows-from-bottom of the piece's lowest and
      // highest cells (pre-clear), rows-from-bottom = 20 - row.
      const maxR = topRow + rot.height - 1;
      const minR = topRow;
      const landingHeight = ((HEIGHT - maxR) + (HEIGHT - minR)) / 2.0;

      let post;
      if (lines) {
        const kept = pre.filter((v) => v !== FULL_ROW);
        post = new Array(lines).fill(0).concat(kept);
      } else {
        post = pre;
      }
      return { post, lines, landingHeight, eroded };
    }

    candidateFeatures() {
      // Enumerate legal placements for the current piece; return placements,
      // their 8-feature vectors, and post-clear afterstate boards.
      const colTop = this._colTop();
      const placements = [];
      const afters = [];
      const feats = [];
      const rots = PIECES[this.current];
      for (let rotIdx = 0; rotIdx < rots.length; rotIdx++) {
        const rot = rots[rotIdx];
        for (let col = 0; col < rot.nPlacements; col++) {
          const topRow = this._dropRow(rot, col, colTop);
          if (topRow < 0) continue;
          const { post, landingHeight, eroded } = this._apply(rot, col, topRow);
          placements.push([rotIdx, col]);
          afters.push(post);
          const board6 = boardFeatures(post);
          feats.push([landingHeight, eroded, ...board6]);
        }
      }
      return { placements, feats, afters };
    }

    step(rotation, column) {
      if (this.gameOver) throw new Error("step() called on a finished game");

      const rots = PIECES[this.current];
      let illegal = !(rotation >= 0 && rotation < rots.length);
      const rot = illegal ? null : rots[rotation];
      if (!illegal && !(column >= 0 && column < rot.nPlacements)) illegal = true;

      let topRow = -1;
      if (!illegal) {
        const colTop = this._colTop();
        topRow = this._dropRow(rot, column, colTop);
        if (topRow < 0) illegal = true;
      }

      if (illegal) {
        this.gameOver = true;
        return {
          linesCleared: 0,
          gameOver: true,
          landingHeight: 0.0,
          eroded: 0,
          features: null,
        };
      }

      const { post, lines, landingHeight, eroded } = this._apply(rot, column, topRow);
      this.rows = post;
      this.lines += lines;
      this.pieces += 1;

      const board6 = boardFeatures(post);
      const features = [landingHeight, eroded, ...board6];

      this.current = this.queue.shift();
      this._fillQueue();
      if (this.legalPlacements().length === 0) this.gameOver = true;

      return {
        linesCleared: lines,
        gameOver: this.gameOver,
        landingHeight,
        eroded,
        features,
      };
    }
  }

  return { TetrisEngine, PIECES };
}
