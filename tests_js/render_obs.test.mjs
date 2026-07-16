// Observation-renderer golden parity tests (PLAN2.md §5, Phase C gate).
//
// For each of the recorded obs fixtures, regenerate the seeded action stream,
// replay it through the JS frame env, render the 96x96 observation at each
// decision tick (before applying that tick's action) with demo/js/render_obs.js,
// and assert the CRC32 (CRC-32/IEEE, matching Python zlib.crc32) of its 9216
// row-major bytes matches the Python fixture bit-exactly.
//
// The action stream is identical to the frame-parity fixtures: a separate
// Mulberry32 seeded with the fixture seed, action = floor(nextFloat() * 5).

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { makeFrameEnv, DECISION_PERIOD } from "../demo/js/frame_env.js";
import { makeRenderObs } from "../demo/js/render_obs.js";
import { Mulberry32 } from "../demo/js/rng.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

const piecesJson = JSON.parse(readFileSync(join(ROOT, "shared", "pieces.json"), "utf8"));
const fixtures = JSON.parse(
  readFileSync(join(ROOT, "shared", "fixtures", "obs_v2.json"), "utf8"),
);

const { FrameEnv } = makeFrameEnv(piecesJson);
const { renderEnv } = makeRenderObs(piecesJson);

// CRC-32/IEEE (reflected, poly 0xEDB88320) — matches Python zlib.crc32.
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

const NUM_ACTIONS = fixtures.num_actions;
const NUM_DECISIONS = fixtures.num_decisions;

assert.equal(fixtures.decision_period, DECISION_PERIOD, "decision period mismatch");
assert.equal(fixtures.obs_size, 96, "obs size mismatch");

// crc32 self-check against a known vector ("123456789" => 0xCBF43926).
test("crc32 matches the standard check vector", () => {
  const bytes = new Uint8Array([..."123456789"].map((ch) => ch.charCodeAt(0)));
  assert.equal(crc32(bytes), 0xcbf43926);
});

for (const fx of fixtures.fixtures) {
  const { seed, crcs } = fx;

  test(`seed ${seed} — observation CRC32 bit-exact (${crcs.length} decision ticks)`, () => {
    const env = new FrameEnv(seed);
    const actionRng = new Mulberry32(seed);
    const got = [];
    while (got.length < NUM_DECISIONS && !env.gameOver) {
      if (env.tickCount % DECISION_PERIOD === 0) {
        got.push(crc32(renderEnv(env)));
        env.apply_action(Math.floor(actionRng.nextFloat() * NUM_ACTIONS));
      }
      env.tick();
    }
    assert.equal(got.length, crcs.length, "observation count mismatch");
    for (let i = 0; i < crcs.length; i++) {
      assert.equal(got[i] >>> 0, crcs[i] >>> 0, `obs CRC mismatch at decision ${i}`);
    }
  });
}
