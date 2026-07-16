// Model History + calibrated-presses wiring: logit-bias helpers, the manifest's
// calibrated default, and the weights-free demo registry index.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  applyLogitBias,
  agentLogitBias,
  argmaxLogits,
  parsePixelManifest,
} from "../demo/js/pixel_agent.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const MANIFEST = join(ROOT, "demo", "models", "manifest.json");
const REGISTRY = join(ROOT, "demo", "models", "registry.json");

test("applyLogitBias: null is a no-op copy", () => {
  const l = [1, 2, 3, 4, 5];
  const out = applyLogitBias(l, null);
  assert.deepEqual(out, l);
  assert.notEqual(out, l); // new array
});

test("applyLogitBias: log-prior suppresses rare-press classes -> noop wins", () => {
  // Raw logits favour a press (index 3); the strong negative press bias flips
  // the argmax back to noop — the anti-spam behaviour.
  const raw = [0.5, 0.2, 0.1, 0.9, 0.0];
  const bias = [-0.08, -4.04, -3.55, -3.9, -5.1];
  assert.equal(argmaxLogits(raw), 3);
  assert.equal(argmaxLogits(applyLogitBias(raw, bias)), 0);
});

test("agentLogitBias: returns a length-5 array or null", () => {
  assert.equal(agentLogitBias({}), null);
  assert.equal(agentLogitBias({ logit_bias: [1, 2, 3] }), null); // wrong length
  assert.deepEqual(agentLogitBias({ logit_bias: [0, 0, 0, 0, 0] }), [0, 0, 0, 0, 0]);
});

test("parsePixelManifest: calibrated default + per-agent logit_bias", {
  skip: existsSync(MANIFEST) ? false : "manifest not built",
}, () => {
  const manifest = JSON.parse(readFileSync(MANIFEST, "utf8"));
  const pm = parsePixelManifest(manifest);
  assert.ok(pm, "pixel block present");
  assert.equal(pm.calibratedDefault, true, "calibration ON by default");
  // Default is the calibrated BC-100 milestone, not the spammy DAgger-final.
  assert.equal(pm.final, "pixel_nn_100");
  const dflt = pm.byId[pm.final];
  assert.ok(agentLogitBias(dflt), "default agent carries a 5-float logit_bias");
  // Honest eval line: presses/piece + thrash surfaced on the agents.
  const dagger = pm.byId["pixel_dagger_1"];
  assert.ok(dagger.eval.presses_per_piece > 10, "raw DAgger-final is spammy");
  assert.ok(dagger.eval.thrash > 0.3, "raw DAgger-final thrashes");
});

test("demo registry.json: weights-free, ordered, full lineage", {
  skip: existsSync(REGISTRY) ? false : "registry.json not exported",
}, () => {
  const doc = JSON.parse(readFileSync(REGISTRY, "utf8"));
  const entries = doc.entries;
  assert.ok(Array.isArray(entries) && entries.length >= 16, "16+ lineage entries");
  const created = entries.map((e) => e.created);
  assert.deepEqual(created, [...created].sort(), "ordered by created date");
  for (const e of entries) {
    assert.equal(e.checkpoint, undefined, "no local weights path in the demo bundle");
    for (const k of ["id", "created", "family", "label", "domain", "eval"]) {
      assert.ok(k in e, `entry ${e.id} missing ${k}`);
    }
  }
  const ids = new Set(entries.map((e) => e.id));
  for (const want of ["v1_td_v1_final", "pixel_bc_v2_100", "pixel_ppo_v2_final"]) {
    assert.ok(ids.has(want), `missing ${want}`);
  }
});
