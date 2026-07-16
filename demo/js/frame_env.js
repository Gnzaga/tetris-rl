// Frame layer with physical-pose locking (PLAN2.md §1, amended).
//
// Bit-exact port of tetris/frame_env.py. The frame layer owns its board (v1
// row-int representation, row 0 top) and applies locks itself at the piece's
// TRUE PHYSICAL POSE — what the camera sees is what locks. Frozen v1 building
// blocks are reused read-only: rotation tables (via makeEngine's PIECES),
// Mulberry32 + SevenBag piece supply (identical queue discipline to the v1
// engine, so a seed yields the v1 piece sequence), and CLEAR_POINTS scoring.
//
// Semantics (frozen §1):
//   * 30 Hz ticks; gravity descends 1 row every GRAVITY_PERIOD (24) ticks.
//   * an action may be emitted every DECISION_PERIOD (3rd) tick (10 Hz).
//   * actions: noop / left / right / rot_cw / rot_ccw. A slide moves 1 column
//     if the destination cells are collision-free, else a silent no-op.
//     Rotation keeps the top-left anchor, clamps column to [0, 10 - width], and
//     fails (silent no-op) on collision. No kicks.
//   * spawn: rot 0, col floor((10 - width) / 2), bbox bottom at board row -1.
//   * lock: on a colliding gravity descent the piece locks at its current
//     physical pose: place cells, clear full rows shifting down, score via
//     CLEAR_POINTS, draw the next piece. Game over iff any locked cell has
//     row < 0 (board left unchanged, mirroring the v1 illegal-step outcome) or
//     the next spawn pose collides.
//
// Tucks: a slide can move a falling piece under an overhang so it rests deeper
// than the v1 straight drop of its (rot, col); the lock resolves at that
// physical pose and the lock event carries a `tuck` flag. v1-consistency
// invariant (fixture-tested): every non-tuck lock transition is bit-identical
// to v1 engine.step(r, c). Runs unmodified in the browser and Node.

import { makeEngine, CLEAR_POINTS } from "./engine.js";
import { WIDTH, HEIGHT, FULL_ROW } from "./features.js";
import { Mulberry32, SevenBag } from "./rng.js";

export const TICK_HZ = 30;
export const GRAVITY_PERIOD = 24;
export const DECISION_PERIOD = 3;

export const ACTIONS = ["noop", "left", "right", "rot_cw", "rot_ccw"];
export const NOOP = 0;
export const LEFT = 1;
export const RIGHT = 2;
export const ROT_CW = 3;
export const ROT_CCW = 4;

export function makeFrameEnv(piecesJson) {
  const { PIECES } = makeEngine(piecesJson);

  class FrameEnv {
    constructor(seed = 0, preview = 5) {
      this.reset(seed, preview);
    }

    reset(seed, preview = 5) {
      this.rng = new Mulberry32(seed);
      this.bag = new SevenBag(this.rng);
      this.preview = preview;
      this.rows = new Array(HEIGHT).fill(0);
      // Piece supply: identical queue discipline to the v1 engine.
      this.queue = [];
      this._fillQueue();
      this.piece = this.queue.shift();
      this._fillQueue();
      this.lines = 0;
      this.pieces = 0;
      this.score = 0; // sum of CLEAR_POINTS[lines] * 100, v1 demo convention
      this.tickCount = 0;
      this.gravityCounter = 0;
      this.gameOver = false;
      this._pending = NOOP;
      this._spawn();
      if (this._collides(this.piece, this.rot, this.col, this.row)) {
        this.gameOver = true;
      }
      return this;
    }

    _fillQueue() {
      while (this.queue.length < this.preview) this.queue.push(this.bag.nextPiece());
    }

    previewPieces() {
      return this.queue.slice(0, this.preview);
    }

    _spawn() {
      const rot0 = PIECES[this.piece][0];
      this.rot = 0;
      this.col = Math.floor((WIDTH - rot0.width) / 2);
      this.row = -rot0.height; // bounding-box bottom at board row -1
    }

    get isDecisionTick() {
      return this.tickCount % DECISION_PERIOD === 0;
    }

    pose() {
      return [this.piece, this.rot, this.col, this.row];
    }

    _collides(piece, rot, col, row) {
      const rows = this.rows;
      for (const [ro, co] of PIECES[piece][rot].cells) {
        const rr = row + ro;
        const cc = col + co;
        if (cc < 0 || cc >= WIDTH) return true;
        if (rr >= HEIGHT) return true;
        if (rr >= 0 && (rows[rr] >> cc) & 1) return true;
      }
      return false;
    }

    _straightDropRow(rotIdx, col) {
      // v1 straight-drop resting top-row for the active piece at (rot, col).
      const rot = PIECES[this.piece][rotIdx];
      let t = HEIGHT;
      for (let pc = 0; pc < rot.width; pc++) {
        const c = col + pc;
        let top = HEIGHT;
        for (let r = 0; r < HEIGHT; r++) {
          if ((this.rows[r] >> c) & 1) {
            top = r;
            break;
          }
        }
        const v = top - 1 - rot.bottom[pc];
        if (v < t) t = v;
      }
      return t;
    }

    apply_action(action) {
      if (!this.isDecisionTick) {
        throw new Error("apply_action is only valid on a decision tick");
      }
      this._pending = action;
    }

    _doAction(action) {
      if (action === NOOP) return;
      if (action === LEFT || action === RIGHT) {
        const nc = this.col + (action === LEFT ? -1 : 1);
        if (!this._collides(this.piece, this.rot, nc, this.row)) this.col = nc;
        return;
      }
      const n = PIECES[this.piece].length;
      const nr =
        action === ROT_CW ? (this.rot + 1) % n : (this.rot - 1 + n) % n;
      const w = PIECES[this.piece][nr].width;
      const nc = Math.min(Math.max(this.col, 0), WIDTH - w);
      if (!this._collides(this.piece, nr, nc, this.row)) {
        this.rot = nr;
        this.col = nc;
      }
    }

    tick() {
      if (this.gameOver) {
        this.tickCount += 1;
        return null;
      }

      if (this.tickCount % DECISION_PERIOD === 0) this._doAction(this._pending);
      this._pending = NOOP;

      let lock = null;
      this.gravityCounter += 1;
      if (this.gravityCounter >= GRAVITY_PERIOD) {
        this.gravityCounter = 0;
        if (this._collides(this.piece, this.rot, this.col, this.row + 1)) {
          lock = this._lockAndSpawn();
        } else {
          this.row += 1;
        }
      }

      this.tickCount += 1;
      return lock;
    }

    _lockAndSpawn() {
      const rotIdx = this.rot;
      const col = this.col;
      const row = this.row;
      const rot = PIECES[this.piece][rotIdx];

      // Straight drop is the highest reachable rest; deeper == tuck.
      const drop = this._straightDropRow(rotIdx, col);
      if (drop > row) {
        throw new Error(
          `straight-drop invariant violated: drop row ${drop} > lock row ` +
            `${row} (piece ${this.piece} rot ${rotIdx} col ${col})`,
        );
      }
      const tuck = drop !== row;

      // Any locked cell above the board => game over; board left unchanged
      // (bit-identical to the v1 illegal-step outcome). The bbox top row
      // always contains a cell, so row < 0 is the exact test.
      if (row < 0) {
        this.gameOver = true;
        return {
          tick: this.tickCount,
          r: rotIdx,
          c: col,
          lines_after: this.lines,
          tuck,
        };
      }

      // Physical lock: place cells, clear full rows shifting above down
      // (identical row surgery to v1 engine._apply).
      for (const [ro, co] of rot.cells) {
        this.rows[row + ro] |= 1 << (col + co);
      }
      let lines = 0;
      for (let r = row; r < row + rot.height; r++) {
        if (this.rows[r] === FULL_ROW) lines += 1;
      }
      if (lines) {
        const kept = this.rows.filter((v) => v !== FULL_ROW);
        this.rows = new Array(lines).fill(0).concat(kept);
      }
      this.lines += lines;
      this.score += CLEAR_POINTS[lines] * 100;
      this.pieces += 1;
      const lock = {
        tick: this.tickCount,
        r: rotIdx,
        c: col,
        lines_after: this.lines,
        tuck,
      };

      // Draw the next piece and spawn; a colliding spawn pose ends the game.
      this.piece = this.queue.shift();
      this._fillQueue();
      this._spawn();
      if (this._collides(this.piece, this.rot, this.col, this.row)) {
        this.gameOver = true;
      }
      return lock;
    }
  }

  return { FrameEnv, PIECES };
}
