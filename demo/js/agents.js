// Placement-selection agents (PLAN.md §4).
//
// Each agent maps an engine's current state to a legal [rotation, column].
// Deterministic given seed / weights. Ports of tetris/agents.py.

import { Mulberry32 } from "./rng.js";

// Dellacherie's published hand weights over the 8 BCTS features (features 7-8
// zeroed), shipped as a built-in agent (PLAN.md §2).
export const DELLACHERIE_WEIGHTS = [-1, 1, -1, -1, -4, -1, 0, 0];

export class RandomAgent {
  constructor(seed = 0, rng = null) {
    this.rng = rng !== null ? rng : new Mulberry32(seed);
  }

  act(engine) {
    const placements = engine.legalPlacements();
    if (placements.length === 0) return null;
    let idx = Math.floor(this.rng.nextFloat() * placements.length);
    if (idx >= placements.length) idx = placements.length - 1; // float==1.0 edge
    return placements[idx];
  }
}

export class LinearAgent {
  constructor(weights) {
    if (weights.length !== 8) {
      throw new Error(`expected 8 weights, got ${weights.length}`);
    }
    this.weights = weights.slice();
  }

  act(engine) {
    const { placements, feats } = engine.candidateFeatures();
    if (placements.length === 0) return null;
    // Tie-break: strict `>` keeps the first maximal index, i.e. the first
    // placement in enumeration order (rotation asc, column asc) — matches
    // np.argmax in the Python LinearAgent.
    let best = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < placements.length; i++) {
      let s = 0;
      const f = feats[i];
      for (let k = 0; k < 8; k++) s += f[k] * this.weights[k];
      if (s > bestScore) {
        bestScore = s;
        best = i;
      }
    }
    return placements[best];
  }
}

export function dellacherieAgent() {
  return new LinearAgent(DELLACHERIE_WEIGHTS);
}
