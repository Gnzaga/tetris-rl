// Deterministic RNG and 7-bag randomizer (PLAN.md §2).
//
// Bit-exact port of the Python `Mulberry32` / `SevenBag`. Every arithmetic step
// is reduced to 32 bits (`>>> 0`, `Math.imul`) so JS and Python draw identical
// piece sequences from a given seed. Runs unmodified in the browser and Node.

export class Mulberry32 {
  constructor(seed) {
    this.state = seed >>> 0;
  }

  nextUint32() {
    let a = (this.state + 0x6d2b79f5) >>> 0;
    this.state = a;
    let t = Math.imul(a ^ (a >>> 15), a | 1) >>> 0;
    t = (((t + Math.imul(t ^ (t >>> 7), t | 61)) >>> 0) ^ t) >>> 0;
    return (t ^ (t >>> 14)) >>> 0;
  }

  nextFloat() {
    // Uniform float in [0, 1): nextUint32() / 2**32.
    return this.nextUint32() / 4294967296;
  }

  clone() {
    const r = new Mulberry32(0);
    r.state = this.state;
    return r;
  }
}

export class SevenBag {
  constructor(rng) {
    this.rng = rng;
    this.bag = [];
  }

  refill() {
    const bag = [0, 1, 2, 3, 4, 5, 6];
    for (let i = 6; i >= 1; i--) {
      const j = Math.floor(this.rng.nextFloat() * (i + 1));
      const tmp = bag[i];
      bag[i] = bag[j];
      bag[j] = tmp;
    }
    this.bag = bag;
  }

  nextPiece() {
    if (this.bag.length === 0) this.refill();
    return this.bag.shift();
  }
}
