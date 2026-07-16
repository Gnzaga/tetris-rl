// Frame layer over the v1 atomic engine (PLAN2.md §1).
//
// Bit-exact port of tetris/frame_env.py. The v1 engine (engine.js) stays ground
// truth for lock / line-clear / game-over; this layer adds real-time semantics:
//   * 30 Hz ticks; gravity descends 1 row every GRAVITY_PERIOD (24) ticks.
//   * an action may be emitted every DECISION_PERIOD (3rd) tick (10 Hz).
//   * actions: noop / left / right / rot_cw / rot_ccw. A slide moves 1 column
//     if the destination cells are collision-free, else a silent no-op.
//     Rotation keeps the top-left anchor, clamps column to [0, 10 - width], and
//     fails (silent no-op) on collision. No kicks.
//   * spawn: rot 0, col floor((10 - width) / 2), bbox bottom at board row -1.
//   * lock: on a colliding gravity descent, defer to engine.step(rot, col) so
//     the board transition is v1-consistent by construction.
//
// Straight-drop invariant: engine drop row <= frame row always (straight drop is
// the highest reachable rest); equality is the no-tuck case. A slide can move a
// falling piece *under* an overhang (a "tuck") so it rests deeper than any v1
// straight drop of its (rot, col) — the lock still defers to engine.step, and the
// lock event carries a `tuck` flag. Runs unmodified in the browser and Node.

import { makeEngine } from "./engine.js";
import { WIDTH, HEIGHT } from "./features.js";

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
  const { TetrisEngine, PIECES } = makeEngine(piecesJson);

  class FrameEnv {
    constructor(seed = 0, preview = 5) {
      this.reset(seed, preview);
    }

    reset(seed, preview = 5) {
      this.engine = new TetrisEngine(seed, preview);
      this.tickCount = 0;
      this.gravityCounter = 0;
      this.gameOver = false;
      this._pending = NOOP;
      this.piece = this.engine.current;
      this._spawn();
      if (this._collides(this.piece, this.rot, this.col, this.row)) {
        this.gameOver = true;
      }
      return this;
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

    get rows() {
      return this.engine.rows;
    }

    get lines() {
      return this.engine.lines;
    }

    get pieces() {
      return this.engine.pieces;
    }

    pose() {
      return [this.piece, this.rot, this.col, this.row];
    }

    _collides(piece, rot, col, row) {
      const rows = this.engine.rows;
      for (const [ro, co] of PIECES[piece][rot].cells) {
        const rr = row + ro;
        const cc = col + co;
        if (cc < 0 || cc >= WIDTH) return true;
        if (rr >= HEIGHT) return true;
        if (rr >= 0 && (rows[rr] >> cc) & 1) return true;
      }
      return false;
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
      const rot = PIECES[this.piece][rotIdx];
      const drop = this.engine._dropRow(rot, col, this.engine._colTop());
      if (drop > this.row) {
        throw new Error(
          `straight-drop invariant violated: engine drop row ${drop} > frame ` +
            `row ${this.row} (piece ${this.piece} rot ${rotIdx} col ${col})`,
        );
      }

      this.engine.step(rotIdx, col);
      const lock = {
        tick: this.tickCount,
        r: rotIdx,
        c: col,
        lines_after: this.engine.lines,
        tuck: drop !== this.row,
      };
      if (this.engine.gameOver) {
        this.gameOver = true;
        return lock;
      }

      this.piece = this.engine.current;
      this._spawn();
      if (this._collides(this.piece, this.rot, this.col, this.row)) {
        this.gameOver = true;
      }
      return lock;
    }
  }

  return { FrameEnv, PIECES };
}
