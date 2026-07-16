// Presentation-only piece controller (PLAN.md §10).
//
// The controller animates a chosen placement (rotate, then translate 1 col/tick,
// then descend 3 rows/tick through the hidden spawn area) purely for display; the
// engine's atomic step(rot,col) is the ONLY thing that mutates game state, so sim
// parity is never affected by animation. On commit it asks the runner for the
// next decision (async — ONNX inference happens here).

import { WIDTH, HEIGHT } from "./engine.js";

export const HIDDEN_ROWS = 3;
const SPAWN_ROW = -2; // top of the piece bounding box while in the hidden area
const SPAWN_COL = 3;

export function colTopOf(rows) {
  const top = new Array(WIDTH).fill(HEIGHT);
  let remaining = WIDTH;
  for (let r = 0; r < HEIGHT && remaining > 0; r++) {
    const v = rows[r];
    if (!v) continue;
    for (let c = 0; c < WIDTH; c++) {
      if (top[c] === HEIGHT && v & (1 << c)) {
        top[c] = r;
        remaining--;
      }
    }
  }
  return top;
}

// Resting bounding-box top row for (rotObj, col) on `rows` — mirrors the engine's
// private _dropRow so ghost/animation land exactly where step() would place them.
export function dropRowOf(rows, rotObj, col) {
  const ct = colTopOf(rows);
  let t = HEIGHT;
  for (let pc = 0; pc < rotObj.width; pc++) {
    const v = ct[col + pc] - 1 - rotObj.bottom[pc];
    if (v < t) t = v;
  }
  return t;
}

export class Controller {
  constructor(engine, runner, pieces, onCommit = () => {}) {
    this.engine = engine;
    this.runner = runner;
    this.PIECES = pieces;
    this.onCommit = onCommit;
    this.decision = null;
    this.anim = null;
    this.deciding = false;
    this.dead = false;
    this.lastDecisionMs = 0;
    // Optional virtual-keypress hook (Phase F keypad overlay). Fires "up" on each
    // animated rotation step and "left"/"right" on each slide step — derived from
    // the display animation only; the engine is never touched. Default no-op so
    // v1 behaviour and the parity tests are unaffected.
    this.onPress = () => {};
  }

  async _run() {
    const t0 = performance.now();
    const d = await this.runner.decide(this.engine);
    this.lastDecisionMs = performance.now() - t0;
    return d;
  }

  // Kick off the first decision. Safe to await.
  async begin() {
    await this._decide();
  }

  async _decide() {
    if (this.engine.gameOver) {
      this.dead = true;
      this.anim = null;
      return;
    }
    this.deciding = true;
    const d = await this._run();
    if (!d) {
      this.dead = true;
      this.deciding = false;
      this.anim = null;
      return;
    }
    this.decision = d;
    const piece = this.engine.current;
    const [rot, col] = d.placement;
    const rots = this.PIECES[piece];
    const targetObj = rots[rot];
    const spawnCol = Math.min(Math.max(0, SPAWN_COL), targetObj.nPlacements - 1);
    this.anim = {
      piece,
      targetRot: rot,
      targetCol: col,
      rotShown: 0,
      rotObj: rots[0],
      colShown: spawnCol,
      rowShown: SPAWN_ROW,
      resting: dropRowOf(this.engine.rows, targetObj, col),
      phase: "rotate",
    };
    this.deciding = false;
  }

  // Advance one animation tick (1× logic step). May commit a placement, which
  // fires the async next-decision (leaving `deciding` true until it resolves).
  tick() {
    if (this.dead || this.deciding || !this.anim) return;
    const a = this.anim;
    const rots = this.PIECES[a.piece];
    if (a.phase === "rotate") {
      if (a.rotShown !== a.targetRot) {
        a.rotShown = (a.rotShown + 1) % rots.length;
        a.rotObj = rots[a.rotShown];
        this.onPress("up");
      }
      if (a.rotShown === a.targetRot) {
        a.rotObj = rots[a.targetRot];
        a.phase = "translate";
      }
      return;
    }
    if (a.phase === "translate") {
      if (a.colShown < a.targetCol) { a.colShown++; this.onPress("right"); }
      else if (a.colShown > a.targetCol) { a.colShown--; this.onPress("left"); }
      if (a.colShown === a.targetCol) a.phase = "descend";
      return;
    }
    if (a.phase === "descend") {
      a.rowShown += 3;
      if (a.rowShown >= a.resting) {
        a.rowShown = a.resting;
        this._commit();
      }
    }
  }

  _commit() {
    const info = this.engine.step(this.anim.targetRot, this.anim.targetCol);
    this.onCommit(info, this.lastDecisionMs);
    this.anim = null;
    this._decide(); // async; sets deciding=true synchronously
  }

  // Commit the in-progress placement immediately (Step control while paused).
  forceCommit() {
    if (this.dead || this.deciding || !this.anim) return;
    this._commit();
  }

  // MAX speed: no animation, decide + step directly. Returns false when dead.
  async stepMax() {
    if (this.dead || this.engine.gameOver) {
      this.dead = true;
      return false;
    }
    this.deciding = true;
    const d = await this._run();
    if (!d) {
      this.dead = true;
      this.deciding = false;
      return false;
    }
    this.decision = d;
    const info = this.engine.step(...d.placement);
    this.deciding = false;
    this.onCommit(info, this.lastDecisionMs);
    this.anim = null;
    if (info.gameOver) {
      this.dead = true;
      return false;
    }
    return true;
  }
}
