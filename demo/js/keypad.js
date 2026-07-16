// Emulator-style keypress overlay (PLAN2.md §8, Phase F).
//
// A D-pad-shaped arrow cluster (↑ ↓ ← → + a centre noop dot) that lights on each
// press, plus a scrolling press tape of the recent keys. Wired to the pixel
// agent's REAL actions (pixel_agent.js) and — in v1 mode — to virtual presses
// derived from the controller's placement animation (rotations → ↑, slides →
// ←/→). Pure DOM; builds its own markup inside a supplied container.

const KEYS = ["up", "down", "left", "right", "noop"];
const GLYPH = { up: "↑", down: "↓", left: "←", right: "→", noop: "•" };
// Tape glyphs bias toward the pixel action legend.
const TAPE_GLYPH = { up: "↑", down: "↓", left: "←", right: "→", noop: "·" };

export class Keypad {
  constructor(container, { tapeLen = 48 } = {}) {
    this.container = container;
    this.tapeLen = tapeLen;
    this.tape = [];
    this._litTimers = {};
    this._build();
  }

  _build() {
    this.container.innerHTML = "";
    this.container.classList.add("keypad");

    const pad = document.createElement("div");
    pad.className = "keypad-grid";
    this.btn = {};
    // 3×3 cluster: up top-centre, left/noop/right middle, down bottom-centre.
    const layout = [
      [null, "up", null],
      ["left", "noop", "right"],
      [null, "down", null],
    ];
    for (const row of layout) {
      for (const cell of row) {
        const el = document.createElement("div");
        if (cell) {
          el.className = `key key-${cell}`;
          el.textContent = GLYPH[cell];
          this.btn[cell] = el;
        } else {
          el.className = "key key-empty";
        }
        pad.appendChild(el);
      }
    }
    this.container.appendChild(pad);

    this.tapeEl = document.createElement("div");
    this.tapeEl.className = "keypad-tape";
    this.container.appendChild(this.tapeEl);
  }

  // key: one of "up","down","left","right","noop".
  press(key) {
    if (!KEYS.includes(key)) return;
    const el = this.btn[key];
    if (el) {
      el.classList.add("lit");
      clearTimeout(this._litTimers[key]);
      this._litTimers[key] = setTimeout(() => el.classList.remove("lit"), 120);
    }
    this.tape.push(key);
    if (this.tape.length > this.tapeLen) this.tape.shift();
    this.tapeEl.textContent = this.tape.map((k) => TAPE_GLYPH[k]).join(" ");
  }

  reset() {
    this.tape = [];
    if (this.tapeEl) this.tapeEl.textContent = "";
    for (const k of KEYS) this.btn[k]?.classList.remove("lit");
  }
}
