# PLAN2.md — Tetris RL v2: Pixel-Input, Keypress-Level, Real-Time Agent

You (Claude Code) are implementing this project end to end, on top of the completed v1 (PLAN.md). Work strictly phase by phase; gates are hard. v1's engine, features, fixtures, trainers, and demo are FROZEN — v2 adds layers, it never modifies v1 semantics. Any file covered by v1 parity fixtures (demo/js/{rng,engine,features,agents}.js, tetris/{rng,engine,features}.py, shared/pieces.json, shared/fixtures/parity_v1.json) must remain byte-compatible with those fixtures.

## 0. Mission

Train an agent that plays Tetris the way a human would have to: it sees only rendered screen pixels, it acts only by pressing arrow keys, and it runs in proportional real time. If the model had a camera and a hand, it could play. Ship it in the existing browser demo with (a) a MarI/O-style live look inside the network and (b) an emulator-style keypress overlay.

Honest expectations (frozen; do not chase v1 numbers): frame-level pixel Tetris is the regime where classical RL fails. Our edge is an immortal v1 teacher enabling unlimited keypress-level demonstrations. Success for v2 = the BC/DAgger agent survives long stretches at level-0 gravity and clears lines consistently (target: median ≥ 100 lines). The pure-RL arm is a time-boxed comparison, expected to lose badly; its purpose is the honest contrast.

## 1. Frozen v2 spec (identical in Python and JS)

**Frame layer.** Wraps the v1 atomic engine (which remains ground truth for lock/clear/game-over). Logic runs at 30 Hz (ticks). Gravity: the active piece descends 1 row every 24 ticks (NES level-0 feel). Decisions: the agent may emit one action every 3rd tick (10 Hz "hand rate"); intermediate ticks advance gravity only.

**Actions (5):** `noop`, `left`, `right`, `rot_cw` (↑), `rot_ccw` (↓). No soft drop, no hard drop, no hold, no DAS. A slide moves the piece 1 column if the destination cells are collision-free; otherwise it is a silent no-op. Rotation: the new rotation state's bounding box keeps the current top-left anchor, column clamped to [0, 10 − width]; if the resulting cells collide with the stack or floor, the rotation fails (silent no-op). No kicks.

**Spawn.** New piece appears at rotation 0, column floor((10 − width)/2), with its bounding box's bottom row at board row −1 (fully above the visible board), descending under gravity. Spawn happens on the tick after lock (no ARE delay). If the spawn pose itself collides (stack reaches above the board), the game is over.

**Lock.** When a gravity descent would collide, the piece locks **at its current physical pose** (what the camera sees is what locks — tucks under overhangs are reachable at 8 decisions per gravity row and must resolve truthfully). The frame layer owns its board (v1 row-int representation) and applies the lock itself: place cells, remove full rows shifting above down, score via clear_points, draw the next piece from the same 7-bag. Game over iff any locked cell has row < 0, or the next spawn pose collides. **v1-consistency invariant (fixture-tested):** whenever the lock pose equals the straight-drop pose for its (rotation, column) — the overwhelmingly common case, and the only case the keypress expert ever produces — the transition must be bit-identical to v1 `engine.step(r, c)`. Lock events record a `tuck` flag.

**Timing invariants.** Tick counter, gravity counter, and decision phase are part of frame-layer state; identical seeds + identical action sequences ⇒ identical tick-by-tick states in Python and JS (fixture-locked, §Phase A gate).

**Observation (the "camera").** Grayscale uint8 canvas, 96×96, rendered at each decision tick; the policy input is a stack of the last 4 observations (40 ms × 3-tick spacing → 400 ms of visual history), normalized to [0,1]. Layout (integer-aligned filled rectangles, no anti-aliasing, no text): board region 10×20 cells at 4 px/cell (40×80 px) with its top-left at (8, 8); a 1 px white border around the board; next-piece preview drawn at 4 px/cell inside a region with top-left (56, 8), 20×20 px, no border. Filled stack cells, active-piece cells, and preview cells all render 255; empty = 0; border = 255. The active piece is NOT visually distinct from the stack — a camera wouldn't distinguish them either. Exact same rasterization in Python (numpy) and JS (drawn on a hidden 96×96 canvas with fillRect only); pixel parity is bit-exact and fixture-locked.

**Episode bookkeeping.** Same 7-bag RNG and scoring as v1. An episode = one game; per-decision reward (for the RL arm) r = clear_points[lines] on the decision at/after the lock tick, −10 terminal; no shaping in v2 (keep the comparison brutal and simple).

## 2. Repo additions (no v1 file moves)

```
tetris/frame_env.py         # frame layer: gravity, keypress transitions, lock→engine.step
tetris/render_obs.py        # 96×96 observation rasterizer (numpy)
tetris/keypress_expert.py   # placement planner → keypress script; reachability by forward sim
tetris/policy_model.py      # PolicyNet CNN (+ named intermediate activation outputs)
tetris/bc.py                # dataset generation, class weighting, BC + DAgger training loops
tetris/ppo.py               # minimal PPO (time-boxed comparison arm), no new deps
scripts/train_bc.py         # BC + DAgger driver (--smoke, --dagger-iters)
scripts/train_ppo.py        # pure-RL arm driver (--smoke, hard time-box)
scripts/gen_fixtures_v2.py  # frame-layer + observation parity fixtures
scripts/export_demo_v2.py   # policy ONNX (multi-output) + manifest v2 additions
demo/js/frame_env.js        # frame layer port (browser + Node)
demo/js/render_obs.js       # observation rasterizer port (hidden canvas)
demo/js/pixel_agent.js      # canvas capture → ONNX policy → key events
demo/js/activations.js      # conv feature-map grids + FC strip + node-wire action head
demo/js/keypad.js           # emulator-style keypress overlay + press tape
tests_js/parity_v2.test.mjs
shared/fixtures/parity_v2.json
```

Same ground rules as PLAN.md §0: no README/docs, --smoke everywhere, determinism everywhere, rich+TB+runio observability for every trainer. Use torch MPS if available (verify numerics vs CPU in smoke); CPU fallback must remain viable with a reduced dataset.

## 3. Phase A — Frame layer + parity

`tetris/frame_env.py` + `demo/js/frame_env.js` per §1. `scripts/gen_fixtures_v2.py` part 1: 15 seeds × scripted pseudo-random action sequences (seeded), 3,000 ticks each, recording tick-by-tick (board hash, piece id, rot, col, row, gravity counter) at every decision tick, plus all lock events with their derived (r, c).

Python tests: gravity/lock/spawn semantics on hand-built scenarios; rotation clamp + failure cases; lock-above-board ⇒ game over; frame-layer determinism; v1-consistency (frame-layer lock sequence replayed through bare v1 engine gives identical boards).

**Gate:** pytest green; `node --test tests_js/*.mjs` green including parity_v2 frame fixtures (15/15).

## 4. Phase B — Keypress expert

`tetris/keypress_expert.py`: for the current spawn, enumerate v1 placements, filter to reachable ones (generate the naive script — rotations, then slides, then waits — and forward-simulate it in the frame env; unreachable if the sim deviates), score reachable afterstates with the td_v1 ValueNet (runs/td_v1 checkpoint; CEM weights fallback via --teacher), emit the chosen script.

**Camera-faithfulness amendment (frozen):** the expert must not act on information the camera cannot see. Its script begins with noops until the active piece is FULLY VISIBLE (every cell at row ≥ 0); rotations/slides start only after that decision tick. The DAgger relabeler obeys the same rule: while the piece is not fully visible, the expert label is `noop`. This costs ≤ 2 gravity descents of fall room; on tall stacks some placements become unreachable — the expert takes the best reachable, and the reachability rate is reported. Expert plays full real-time games headlessly (ticks simulated, not wall-clocked).

**Gates:** (1) pytest green (script validity: forward-sim always lands the predicted (r,c)); (2) expert real-time eval, 20 games, 10k-piece cap, fixed seeds: median lines ≥ 50% of td_v1's capped median (i.e., ≥ ~2,000). Report the reachability rate (fraction of v1-optimal placements reachable; expect ≈ 1.0 at level-0).

## 5. Phase C — Observation renderer + dataset

`tetris/render_obs.py` + `demo/js/render_obs.js` per §1. `gen_fixtures_v2.py` part 2: observation fixtures — for 5 seeds × 50 decision ticks, store CRC32 of the 96×96 buffer; JS must match bit-exactly.

Dataset: expert plays ~25k pieces (~3M decision frames) across seeded games; store (obs_stack, action) with obs as packed bits or uint8 memmap (~28 GB raw is too big — store single 96×96 frames once and reconstruct stacks by index; ~850 MB uint8, acceptable in runs/). Record class histogram (noop will dominate ~90%+); dataset writer computes inverse-frequency class weights, capped at 20×.

**Gates:** pytest green (round-trip, stack reconstruction); JS observation fixtures bit-exact; a linear probe trained on 10k frames recovers per-column stack heights from pixels with ≤ 0.5 mean absolute error (proves the render carries the state).

## 6. Phase D — PolicyNet + BC (+ DAgger)

`tetris/policy_model.py` — PolicyNet: input [B, 4, 96, 96] → Conv(4→16, 8×8, stride 4)+ReLU → Conv(16→32, 4×4, stride 2)+ReLU → Conv(32→32, 3×3, stride 1)+ReLU → Flatten → FC(→256)+ReLU → FC(256→5) logits (+ separate value head FC(256→1) for PPO reuse). Forward exposes named intermediate activations (conv1, conv2, conv3, fc, logits) for the demo; ONNX export (Phase F prep) emits all of them as named outputs.

`tetris/bc.py` + `scripts/train_bc.py`: **class-balanced batch sampling** (each batch drawn ~uniformly over the 5 action classes via a per-class index sampler; noop's natural ~98% share must not dominate batches) with unweighted cross-entropy (or mild residual weights — document), Adam 3e-4, batch 256, ~4 epochs' worth of optimizer steps over the balanced stream; eval every epoch fraction: 20 closed-loop real-time games (greedy argmax), fixed seeds, 10k-piece cap; runio logging (loss, eval_median_lines, accuracy per class); milestone checkpoints 0/25/50/100% of optimizer steps. Then `--dagger-iters N` (default 2): roll out the student for ~300k decision frames, relabel every frame with the expert's action, aggregate, retrain.

**Covariate-shift amendment (frozen, post-mortem-driven):** plain BC + 2×DAgger provably fails here (median 0; see the Phase D debug report — all plumbing verified correct, agreement on self-visited states 0.25). The primary dataset is therefore generated DART-style: the data-collection policy is the expert with **noise injection** (each decision, with probability p drawn per-episode from {0.05, 0.10, 0.20}, replace the expert's action with a uniformly random one), and EVERY visited state is labeled with the current-pose replan action (the same relabeler DAgger uses — itself verified to score ~118 lines as a policy). This bakes recovery states into the base distribution. Batch composition: 50% noop / 50% presses (softer than 5-way uniform — the uniform balance caused false-press thrashing). DAgger iterations remain available on top; the gate (median ≥ 100) is unchanged.

**Gates:** (1) --smoke (<2 min, tiny net + tiny dataset) green, run dir well-formed; (2) full BC+DAgger run `bc_v2`: closed-loop median ≥ 100 lines (20 games, fixed seeds); (3) monotonic-ish: DAgger final ≥ BC-only ≥ 25%-checkpoint. Report MPS vs CPU choice, measured throughput, and wall-clock before launching the full run; if projected > 12 h, stop and report.

## 7. Phase E — Pure-RL comparison arm (time-boxed)

`tetris/ppo.py` + `scripts/train_ppo.py`: minimal PPO-clip, GAE(λ=0.95), γ=0.99, entropy bonus 0.01, 16 parallel frame envs, rollout 128 decisions/env, 4 epochs/batch, lr 2.5e-4, same PolicyNet from scratch. Reward per §1 (clear_points at lock, −10 terminal, nothing else). HARD time-box: `--max-hours 4` (or 5M decision frames, whichever first) — the trainer exits cleanly at the box and writes its final checkpoint/eval regardless of performance.

**Gates:** (1) --smoke green; (2) full run `ppo_v2` completes within the box and logs the same eval protocol as Phase D. There is NO performance gate — the result is reported as-is (expected: near-zero lines; that contrast is the point). If it happens to learn, report that too.

## 8. Phase F — Export + demo integration

`scripts/export_demo_v2.py`: export bc_v2 final (and its 0/25/50/100% milestones) + ppo_v2 final to multi-output ONNX (opset 17, dynamic batch); parity < 1e-4 on logits AND all tapped activations over 1,000 random obs stacks; extend demo/models/manifest.json with a `pixel_agents` section (id, label, path, eval stats, activation output names, action legend) — v1 manifest keys unchanged (v1 demo must keep working with the extended file).

Demo (new "Pixel Agent" mode in the existing page, same dark theme, no new deps):
- Runs frame_env.js at true 30 Hz wall-clock (1×; optional 4× fast-forward). MAX is not offered — v2 is real-time by definition.
- render_obs.js draws the observation each decision tick; pixel_agent.js feeds the stacked tensor to ORT, argmax → key event into the frame env. Board is rendered by the existing big canvas as usual; a "model's-eye" 96×96 inset shows exactly what the network sees.
- **Activation view (toggle):** conv1/conv2/conv3 feature-map grids (tiny heatmaps, updated each inference), FC-256 activation strip, and a node-and-wire graph for FC→5 head (edge thickness/brightness = weight × activation, winning action node glows). Canvas-drawn, throttleable to every Nth inference if frame budget demands.
- **Keypress overlay:** emulator-style arrow pad lighting on each press + scrolling press tape. Wired to real v2 actions; also enabled for v1 agents by deriving virtual presses from the existing controller animation steps (rotations → ↑, slides → ←/→) — v1 engine untouched.
- Self-test extension: 2 fixed obs stacks through the final pixel ONNX, compare logits to manifest values within 1e-3, footer pass/fail alongside the v1 self-test.

**Gates:** (1) all pytest + both node parity suites green; (2) export parity green incl. activation outputs; (3) manifest v2 schema pytest; (4) human visual checklist: pixel agent plays in real time, model's-eye inset live, activations animate, keypad lights match visible piece behavior, v1 demo features all still work, offline OK.

## 9. Phase G — Integration + handoff

Fresh-clone dry run extended: v1 sequence + `train_bc.py --smoke` + `train_ppo.py --smoke` + `export_demo_v2.py` (smoke models) + serve + both self-tests pass. Final report must include: BC vs DAgger vs PPO eval table, the honest PPO contrast, wall-clocks, and the v2 runbook.

## 10. Definition of done (v2)

- [ ] Frame-layer + observation parity fixtures green in both engines (15/15, bit-exact obs).
- [ ] Keypress expert ≥ 50% of td_v1 capped median in real-time play; reachability reported.
- [ ] bc_v2 (BC+DAgger): closed-loop median ≥ 100 lines; milestones exported.
- [ ] ppo_v2: completed within time-box, honestly reported.
- [ ] Multi-output ONNX parity < 1e-4 (logits + activations); demo self-tests pass.
- [ ] Demo: pixel agent real-time, model's-eye inset, activation view with node-wire action head, keypress overlay (v2 + v1), all v1 features intact, offline.
- [ ] v1 fixtures still green, byte-identical v1 parity files.

## 11. Out of scope for v2

Gravity speed curriculum / higher levels · DAS/ARR key repeat · soft/hard drop · lock delay, kicks · color observations · camera noise/jitter augmentation · sound · deploying beyond local.
