"""Behavioral-cloning dataset generation + class weighting (PLAN2.md §5, Phase C).

This module currently holds the Phase C **dataset half** only: generating the
keypress-expert demonstration corpus, packing observations to disk, and the
class histogram / inverse-frequency weights. The BC + DAgger **training loops**
are Phase D and will be added to this same file (per the PLAN2.md §2 layout).

Storage format (runs/bc_data_v1/ by default; runs/ is gitignored)
------------------------------------------------------------------
The expert plays seeded frame-layer games; at every decision tick we store the
96x96 observation the agent saw (rendered BEFORE its action) plus the action it
chose. Observations take three values ({0, 128, 255}; §1 perception amendment),
so each frame is bit-packed into TWO :func:`numpy.packbits` bitplanes — plane 0
= "filled" (obs != 0), plane 1 = "bright" (obs == 255) — for 2304 bytes/frame
(2 * 96*96 / 8), reconstructed as ``filled*128 + bright*127``. That is a ~4x
saving over raw uint8 (which would be ~35 GB for ~3.8M frames). Files:

* ``frames.u8pack`` : raw ``(N, 2304)`` uint8 memmap of packed frames.
* ``actions.npy``   : ``(N,)`` uint8 action labels (0..4).
* ``episode_id.npy``: ``(N,)`` int32 episode index per frame.
* ``meta.json``     : counts, per-episode table, class histogram + weights,
  pixel layout, and the exact packing/stack conventions.

4-stacks are reconstructed by index and never cross an episode boundary: the
stack for frame ``i`` (in the episode starting at ``s``) is
``[max(i-3, s), max(i-2, s), max(i-1, s), i]`` — the first frames of an episode
repeat that episode's first frame (documented, matches the closed-loop policy's
cold start).

Class weights: inverse-frequency, normalized so the majority class (noop) has
weight 1.0, then clipped to a maximum of 20x (PLAN2.md §5).
"""

from __future__ import annotations

import json
from array import array
from pathlib import Path

import numpy as np

from tetris.features import HEIGHT, WIDTH
from tetris.frame_env import ACTIONS, FrameEnv
from tetris.keypress_expert import ExpertPlayer, make_teacher
from tetris.render_obs import OBS_SIZE, render_env

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = _REPO / "runs" / "bc_data_v1"

NUM_ACTIONS = len(ACTIONS)  # 5
PLANE_BYTES = OBS_SIZE * OBS_SIZE // 8  # 1152 (one bitplane)
# Two bitplanes per frame since the §1 perception amendment (obs values are
# {0, 128, 255}): plane 0 = "filled at all" (value != 0), plane 1 = "bright"
# (value == 255). Gray active-piece pixels are filled-but-not-bright.
PACKED_BYTES = 2 * PLANE_BYTES  # 2304
AUX_IGNORE = 255  # aux-target mask value (frame has no visible-piece plan)
WEIGHT_CAP = 20.0  # inverse-frequency weights clipped to this multiple of noop


# --------------------------------------------------------------------------
# Obs <-> packed-bits helpers
# --------------------------------------------------------------------------


def pack_obs(obs: np.ndarray) -> np.ndarray:
    """(96,96) uint8 {0,128,255} -> (2304,) uint8: two packed bitplanes
    (numpy MSB-first) — [filled = obs != 0 | bright = obs == 255]."""
    flat = obs.reshape(-1)
    return np.concatenate([np.packbits(flat != 0), np.packbits(flat == 255)])


def unpack_obs(packed: np.ndarray) -> np.ndarray:
    """(2304,) packed bitplanes -> (96,96) uint8 {0,128,255} (inverse of
    :func:`pack_obs`)."""
    packed = np.asarray(packed, dtype=np.uint8)
    filled = np.unpackbits(packed[:PLANE_BYTES])
    bright = np.unpackbits(packed[PLANE_BYTES:])
    # bright implies filled; filled-only pixels are the gray active piece.
    return (filled * 128 + bright * 127).astype(np.uint8).reshape(OBS_SIZE, OBS_SIZE)


def column_heights(rows) -> np.ndarray:
    """(10,) int array: locked-stack height of each column (HEIGHT - topmost
    filled row; 0 for an empty column). Ground-truth probe label."""
    h = np.zeros(WIDTH, dtype=np.int64)
    for c in range(WIDTH):
        for r in range(HEIGHT):
            if (rows[r] >> c) & 1:
                h[c] = HEIGHT - r
                break
    return h


# --------------------------------------------------------------------------
# Class histogram + inverse-frequency weights
# --------------------------------------------------------------------------


def class_histogram(actions: np.ndarray) -> np.ndarray:
    """(5,) int64 count of each action label."""
    return np.bincount(np.asarray(actions, dtype=np.int64), minlength=NUM_ACTIONS)


def inverse_freq_weights(hist: np.ndarray, cap: float = WEIGHT_CAP) -> np.ndarray:
    """Inverse-frequency class weights normalized so the MOST COMMON class has
    weight 1.0, then clipped to ``cap`` (PLAN2.md §5: capped 20x). Absent classes
    get the cap (they never contribute to the loss anyway)."""
    hist = np.asarray(hist, dtype=np.float64)
    with np.errstate(divide="ignore"):
        inv = hist.max() / hist  # majority class -> 1.0
    inv[~np.isfinite(inv)] = cap
    return np.clip(inv, 1.0, cap)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _play_episode(player: ExpertPlayer, seed: int, max_pieces: int,
                  frames_f, actions: array, episode_ids: array,
                  ep_index: int) -> tuple[int, int, int]:
    """Play one expert game to game-over or ``max_pieces``, streaming each
    decision-tick (packed obs -> ``frames_f``, action -> ``actions``,
    ``ep_index`` -> ``episode_ids``). Returns (n_frames, lines, pieces)."""
    env = FrameEnv(seed=seed)
    player.reset(env)
    n = 0
    while not env.game_over and env.pieces < max_pieces:
        if env.is_decision_tick:
            obs = render_env(env)  # what the agent sees, before it acts
            frames_f.write(pack_obs(obs).tobytes())
            action = player.act(env)
            actions.append(action)
            episode_ids.append(ep_index)
            env.apply_action(action)
            n += 1
        env.tick()
    return n, env.lines, env.pieces


def generate_dataset(
    out_dir: str | Path = DEFAULT_DATA_DIR,
    total_pieces: int = 25000,
    max_game_pieces: int = 1000,
    base_seed: int = 100000,
    teacher_kind: str = "td",
    checkpoint=None,
    device: str = "cpu",
    progress: bool = True,
) -> dict:
    """Generate the BC demonstration dataset (module docstring). Plays seeded
    expert games (``base_seed`` + episode index) until ``total_pieces`` is
    reached, capping each game at ``max_game_pieces`` for board diversity.
    Returns the meta dict (also written to ``meta.json``)."""
    import time

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.u8pack"

    teacher = make_teacher(teacher_kind, checkpoint, device)
    player = ExpertPlayer(teacher)

    actions = array("B")
    episode_ids = array("i")
    episodes = []  # (start, length, seed, lines, pieces)

    n = 0
    pieces_total = 0
    ep = 0
    t0 = time.perf_counter()
    with open(frames_path, "wb", buffering=1 << 20) as frames_f:
        while pieces_total < total_pieces:
            seed = base_seed + ep
            start = n
            cnt, lines, pieces = _play_episode(
                player, seed, max_game_pieces, frames_f, actions, episode_ids, ep
            )
            episodes.append((start, cnt, seed, int(lines), int(pieces)))
            n += cnt
            pieces_total += pieces
            ep += 1
            if progress:
                el = time.perf_counter() - t0
                print(
                    f"episode {ep:>3} seed={seed} pieces={pieces:>5} lines={lines:>6} "
                    f"frames={cnt:>6} | total: {pieces_total:>6}/{total_pieces} pieces, "
                    f"{n:>8} frames, {el:6.1f}s",
                    flush=True,
                )
    elapsed = time.perf_counter() - t0

    actions_arr = np.frombuffer(actions, dtype=np.uint8).copy()
    episode_arr = np.frombuffer(episode_ids, dtype=np.int32).copy()
    np.save(out_dir / "actions.npy", actions_arr)
    np.save(out_dir / "episode_id.npy", episode_arr)

    hist = class_histogram(actions_arr)
    weights = inverse_freq_weights(hist)

    meta = {
        "n_frames": int(n),
        "n_episodes": int(ep),
        "total_pieces": int(pieces_total),
        "total_lines": int(sum(e[3] for e in episodes)),
        "obs_size": OBS_SIZE,
        "packed_bytes": PACKED_BYTES,
        "num_actions": NUM_ACTIONS,
        "actions": list(ACTIONS),
        "teacher": teacher_kind,
        "base_seed": base_seed,
        "max_game_pieces": max_game_pieces,
        "packing": (
            "two MSB-first bitplanes [filled = obs != 0 | bright = obs == 255], "
            "2304 bytes/frame; reconstruct obs {0,128,255} as filled*128 + bright*127"
        ),
        "stack_rule": (
            "4-stack for frame i in episode starting s = "
            "[max(i-3,s), max(i-2,s), max(i-1,s), i]; episode-start frames repeat"
        ),
        "class_histogram": {ACTIONS[a]: int(hist[a]) for a in range(NUM_ACTIONS)},
        "class_fractions": {
            ACTIONS[a]: (float(hist[a] / n) if n else 0.0) for a in range(NUM_ACTIONS)
        },
        "class_weights": {ACTIONS[a]: float(weights[a]) for a in range(NUM_ACTIONS)},
        "weight_cap": WEIGHT_CAP,
        "elapsed_s": round(elapsed, 2),
        "episodes": episodes,
        "files": {
            "frames": "frames.u8pack",
            "actions": "actions.npy",
            "episode_id": "episode_id.npy",
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if progress:
        size_gb = frames_path.stat().st_size / 1e9
        print(
            f"\nDONE: {n} frames, {ep} episodes, {pieces_total} pieces, "
            f"{meta['total_lines']} lines in {elapsed:.1f}s "
            f"({n / elapsed:.0f} frames/s); frames.u8pack = {size_gb:.2f} GB"
        )
        print("class histogram:", meta["class_histogram"])
        print("class weights  :", {k: round(v, 3) for k, v in meta["class_weights"].items()})
    return meta


# --------------------------------------------------------------------------
# DART-style noise-injected generation (PLAN2.md §6 covariate-shift amendment)
# --------------------------------------------------------------------------
#
# Plain BC + 2xDAgger provably fails closed-loop here (Phase D debug report:
# agreement on self-visited states 0.25 despite 0.99 held-in accuracy). The
# primary dataset is therefore collected DART-style: the behavior policy is the
# current-pose replan relabeler (itself verified to score ~118 lines as a
# policy) with per-decision noise injection — with probability p (drawn per
# episode from DART_NOISE_LEVELS) the APPLIED action is replaced by a uniform
# random one — while the stored LABEL is always the relabeler's action for the
# visited state. Recovery states (piece off the optimal path) are thereby baked
# into the base distribution with correct corrective labels. Labels remain
# camera-faithful: the relabeler returns NOOP while the piece is not fully
# visible (§4), even when noise displaced it up there.

DART_NOISE_LEVELS = (0.05, 0.10, 0.20)


def _play_dart_episode(relabeler, seed: int, noise_p: float, noise_rng,
                       max_pieces: int, frames_f, actions: array,
                       episode_ids: array, ep_index: int, targets: array):
    """One noise-injected relabeler-policy game, streamed like
    :func:`_play_episode`. Also records the expert plan's aux target
    (rot, col) per frame (AUX_IGNORE pair when undefined). Returns
    (n_frames, lines, pieces, n_noised, n_press_labels)."""
    env = FrameEnv(seed=seed)
    n = noised = presses = 0
    while not env.game_over and env.pieces < max_pieces:
        if env.is_decision_tick:
            obs = render_env(env)  # what the agent sees, before it acts
            frames_f.write(pack_obs(obs).tobytes())
            label, t_rot, t_col = relabeler.relabel_with_target(env)
            actions.append(label)
            episode_ids.append(ep_index)
            targets.append(t_rot)
            targets.append(t_col)
            if label != 0:
                presses += 1
            applied = label
            if noise_rng.random() < noise_p:
                applied = int(noise_rng.integers(0, NUM_ACTIONS))
                noised += 1
            env.apply_action(applied)
            n += 1
        env.tick()
    return n, env.lines, env.pieces, noised, presses


def generate_dart_dataset(
    out_dir: str | Path,
    total_pieces: int = 25000,
    max_game_pieces: int = 1000,
    base_seed: int = 200000,
    noise_seed: int = 7,
    noise_levels: tuple = DART_NOISE_LEVELS,
    teacher_kind: str = "td",
    checkpoint=None,
    device: str = "cpu",
    progress: bool = True,
) -> dict:
    """Generate the DART noise-injected demonstration dataset (block comment
    above). Same on-disk format as :func:`generate_dataset` (readable by
    :class:`BCDataset`); fully deterministic given the arguments — episode seeds
    are ``base_seed + ep`` and all noise draws come from a single
    ``np.random.default_rng(noise_seed)`` stream consumed in episode order."""
    import time

    from tetris.keypress_expert import DaggerRelabeler

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.u8pack"

    teacher = make_teacher(teacher_kind, checkpoint, device)
    noise_rng = np.random.default_rng(noise_seed)

    actions = array("B")
    episode_ids = array("i")
    targets = array("B")  # flat (rot, col) pairs, AUX_IGNORE where undefined
    episodes = []  # (start, length, seed, lines, pieces, noise_p, n_noised)

    n = 0
    pieces_total = 0
    press_total = 0
    ep = 0
    t0 = time.perf_counter()
    with open(frames_path, "wb", buffering=1 << 20) as frames_f:
        while pieces_total < total_pieces:
            seed = base_seed + ep
            p = float(noise_levels[int(noise_rng.integers(len(noise_levels)))])
            start = n
            cnt, lines, pieces, noised, presses = _play_dart_episode(
                DaggerRelabeler(teacher), seed, p, noise_rng,
                max_game_pieces, frames_f, actions, episode_ids, ep, targets,
            )
            episodes.append((start, cnt, seed, int(lines), int(pieces), p, noised))
            n += cnt
            pieces_total += pieces
            press_total += presses
            ep += 1
            if progress:
                el = time.perf_counter() - t0
                print(
                    f"dart ep {ep:>3} seed={seed} p={p:.2f} pieces={pieces:>5} "
                    f"lines={lines:>6} frames={cnt:>6} noised={noised:>5} | "
                    f"total: {pieces_total:>6}/{total_pieces} pieces, {n:>8} frames, "
                    f"press_frac={press_total / max(n, 1):.4f}, {el:6.1f}s "
                    f"({n / max(el, 1e-9):.0f} f/s)",
                    flush=True,
                )
    elapsed = time.perf_counter() - t0

    actions_arr = np.frombuffer(actions, dtype=np.uint8).copy()
    episode_arr = np.frombuffer(episode_ids, dtype=np.int32).copy()
    targets_arr = np.frombuffer(targets, dtype=np.uint8).copy().reshape(-1, 2)
    np.save(out_dir / "actions.npy", actions_arr)
    np.save(out_dir / "episode_id.npy", episode_arr)
    np.save(out_dir / "targets.npy", targets_arr)

    hist = class_histogram(actions_arr)
    meta = {
        "n_frames": int(n),
        "n_episodes": int(ep),
        "total_pieces": int(pieces_total),
        "total_lines": int(sum(e[3] for e in episodes)),
        "obs_size": OBS_SIZE,
        "packed_bytes": PACKED_BYTES,
        "num_actions": NUM_ACTIONS,
        "actions": list(ACTIONS),
        "source": "dart",
        "targets": "targets.npy (N,2) uint8 (rot, col); 255 = undefined/masked",
        "aux_target_coverage": float((targets_arr[:, 0] != AUX_IGNORE).mean()) if n else 0.0,
        "behavior_policy": "current-pose replan relabeler + uniform noise",
        "noise_levels": list(noise_levels),
        "noise_seed": noise_seed,
        "teacher": teacher_kind,
        "base_seed": base_seed,
        "max_game_pieces": max_game_pieces,
        "packing": (
            "two MSB-first bitplanes [filled = obs != 0 | bright = obs == 255], "
            "2304 bytes/frame; reconstruct obs {0,128,255} as filled*128 + bright*127"
        ),
        "stack_rule": (
            "4-stack for frame i in episode starting s = "
            "[max(i-3,s), max(i-2,s), max(i-1,s), i]; episode-start frames repeat"
        ),
        "class_histogram": {ACTIONS[a]: int(hist[a]) for a in range(NUM_ACTIONS)},
        "class_fractions": {
            ACTIONS[a]: (float(hist[a] / n) if n else 0.0) for a in range(NUM_ACTIONS)
        },
        "press_fraction": float((n - hist[0]) / n) if n else 0.0,
        "elapsed_s": round(elapsed, 2),
        "episodes": episodes,
        "files": {
            "frames": "frames.u8pack",
            "actions": "actions.npy",
            "episode_id": "episode_id.npy",
            "targets": "targets.npy",
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if progress:
        print(f"\nDART DONE: {n} frames, {ep} episodes, {pieces_total} pieces, "
              f"{meta['total_lines']} lines in {elapsed:.1f}s; "
              f"press_fraction={meta['press_fraction']:.4f}")
        print("class histogram:", meta["class_histogram"])
    return meta


# --------------------------------------------------------------------------
# Dataset reader (used by Phase C tests and, later, Phase D training)
# --------------------------------------------------------------------------


class BCDataset:
    """Random-access reader over a generated dataset. Reconstructs normalized
    4-stacks by index (never crossing an episode boundary)."""

    def __init__(self, out_dir: str | Path = DEFAULT_DATA_DIR):
        out_dir = Path(out_dir)
        with open(out_dir / "meta.json") as f:
            self.meta = json.load(f)
        if int(self.meta["packed_bytes"]) != PACKED_BYTES:
            raise ValueError(
                f"{out_dir}: packed_bytes {self.meta['packed_bytes']} != "
                f"{PACKED_BYTES} — dataset predates the two-bitplane format "
                f"(§1 perception amendment); regenerate it")
        self.n = int(self.meta["n_frames"])
        self.frames = np.memmap(
            out_dir / "frames.u8pack", dtype=np.uint8, mode="r",
            shape=(self.n, PACKED_BYTES),
        )
        self.actions = np.load(out_dir / "actions.npy")
        self.episode_id = np.load(out_dir / "episode_id.npy")
        # Aux target labels (rot, col) per frame; AUX_IGNORE where undefined
        # (not fully visible / no reachable plan / legacy corpus without them).
        tpath = out_dir / "targets.npy"
        self.targets = (np.load(tpath) if tpath.exists()
                        else np.full((self.n, 2), AUX_IGNORE, dtype=np.uint8))
        # Per-frame episode-start index for O(1) stack reconstruction.
        self.ep_start = np.empty(self.n, dtype=np.int64)
        for start, length, *_ in self.meta["episodes"]:
            self.ep_start[start:start + length] = start

    def __len__(self) -> int:
        return self.n

    def frame(self, i: int) -> np.ndarray:
        """(96,96) uint8 {0,128,255} observation at index ``i``."""
        return unpack_obs(self.frames[i])

    def stack_indices(self, i: int) -> list[int]:
        s = int(self.ep_start[i])
        return [max(i - 3, s), max(i - 2, s), max(i - 1, s), i]

    def _frames_float(self, flat: np.ndarray) -> np.ndarray:
        """Unpack frames at ``flat`` indices -> (K, 96, 96) float32, values in
        {0, 128/255, 1} — bit-identical to the closed-loop uint8/255.0 path."""
        bits = np.unpackbits(self.frames[flat], axis=1)  # (K, 18432)
        npx = OBS_SIZE * OBS_SIZE
        filled = bits[:, :npx].astype(np.float32)
        bright = bits[:, npx:].astype(np.float32)
        vals = (filled * 128.0 + bright * 127.0) / 255.0
        return vals.reshape(len(flat), OBS_SIZE, OBS_SIZE)

    def stack(self, i: int) -> np.ndarray:
        """(4,96,96) float32 in [0,1]: the last-4-observation policy input."""
        idx = np.asarray(self.stack_indices(i), dtype=np.int64)
        return self._frames_float(idx)

    def batch_stacks(self, idx: np.ndarray) -> np.ndarray:
        """Vectorized :meth:`stack` for a batch of indices -> (B,4,96,96) float32
        in {0,1}. Reconstructs each 4-stack by index (episode-boundary safe) and
        bit-unpacks all B*4 source frames in one numpy op — the BC training hot
        path never touches per-sample python."""
        idx = np.asarray(idx, dtype=np.int64)
        s = self.ep_start[idx]
        src = np.stack(
            [np.maximum(idx - 3, s), np.maximum(idx - 2, s),
             np.maximum(idx - 1, s), idx],
            axis=1,
        )  # (B, 4)
        flat = src.reshape(-1)
        return self._frames_float(flat).reshape(len(idx), 4, OBS_SIZE, OBS_SIZE)


# ==========================================================================
# Phase D — BC + DAgger training / closed-loop policy eval (PLAN2.md §6)
# ==========================================================================
#
# Everything below is the training half of this module. The dataset half above
# is torch-free; these functions import torch lazily so `import tetris.bc` in a
# torch-free context (e.g. CEM workers) stays cheap.
#
# Metrics-schema note (frozen `tetris.runio` METRICS_FIELDS): closed-loop line
# stats go into metrics.jsonl via RunWriter.log; PER-CLASS ACCURACY is not a
# schema field, so it is carried in each checkpoint's payload metadata (and
# printed) instead — the schema stays frozen. For BC/DAgger the `pieces_trained`
# metrics/TB x-axis carries the OPTIMIZER-STEP index (documented in config).

import math
import statistics
from collections import deque

from tetris.evaluation import p10
from tetris.keypress_expert import DaggerRelabeler, make_teacher
from tetris.policy_model import NUM_ACTIONS as _NPI  # noqa: F401  (== NUM_ACTIONS)
from tetris.policy_model import PolicyNet

EVAL_SEED_BASE = 950000  # closed-loop eval seeds (NOT the dataset seeds)


# --------------------------------------------------------------------------
# Device selection + MPS/CPU numeric parity
# --------------------------------------------------------------------------


def select_device(prefer: str = "mps") -> str:
    """Return ``"mps"`` if requested and available, else ``"cpu"``."""
    import torch

    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer not in ("mps", "cpu"):
        return prefer
    return "cpu"


def mps_cpu_max_logit_diff(model: "PolicyNet", device: str, seed: int = 0) -> float:
    """Max abs difference between CPU and ``device`` policy logits on one random
    batch (same weights). Returns 0.0 when ``device`` is cpu. Used by the smoke
    gate to verify MPS numerics (< 1e-3) before trusting the accelerator."""
    import torch

    if device == "cpu":
        return 0.0
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(32, 4, OBS_SIZE, OBS_SIZE, generator=g)
    cpu_model = PolicyNet()
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.eval()
    dev_model = PolicyNet().to(device)
    dev_model.load_state_dict(model.state_dict())
    dev_model.eval()
    with torch.no_grad():
        lc, _ = cpu_model(x)
        ld, _ = dev_model(x.to(device))
    return float((lc - ld.cpu()).abs().max().item())


# --------------------------------------------------------------------------
# Multi-dataset view (base corpus + DAgger relabels), same stack semantics
# --------------------------------------------------------------------------


class MultiBCDataset:
    """Concatenated view over several :class:`BCDataset` shards. Global indices
    map to (shard, local); 4-stacks are reconstructed WITHIN each shard so
    episode boundaries are always respected. ``actions`` is the concatenated
    label array; :meth:`batch_stacks` dispatches a batch of global indices to the
    right shard(s)."""

    def __init__(self, datasets: list[BCDataset]):
        if not datasets:
            raise ValueError("MultiBCDataset needs >= 1 shard")
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)]).astype(np.int64)
        self.n = int(self.offsets[-1])
        self.actions = np.concatenate([d.actions for d in datasets])
        self.targets = np.concatenate([d.targets for d in datasets])

    def __len__(self) -> int:
        return self.n

    def batch_stacks(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((len(idx), 4, OBS_SIZE, OBS_SIZE), dtype=np.float32)
        which = np.searchsorted(self.offsets, idx, side="right") - 1
        for di, d in enumerate(self.datasets):
            m = which == di
            if m.any():
                out[m] = d.batch_stacks(idx[m] - self.offsets[di])
        return out


# --------------------------------------------------------------------------
# Training primitives
# --------------------------------------------------------------------------


def num_optimizer_steps(n: int, batch_size: int, epochs: int) -> int:
    """"``epochs`` epochs' WORTH" of optimizer steps: the balanced sampler (below)
    has no natural epoch boundary, so the step budget is defined as the count a
    plain shuffled pass would have used — ``epochs * ceil(n / batch)`` — keeping
    the budget comparable to the pre-amendment weighted-CE run."""
    return epochs * math.ceil(n / batch_size)


def balanced_batches(dataset, batch_size: int, total_steps: int,
                     rng: np.random.Generator):
    """Class-balanced batch stream (PLAN2.md §6 amendment). Yields exactly
    ``total_steps`` ``(stacks[B,4,96,96] f32, labels[B] int64)`` batches, each
    drawn ~uniformly over the action classes PRESENT in ``dataset``: every class
    contributes ``batch // k`` samples (the remainder spread one-per-class), so
    noop's natural ~98% share never dominates a batch. Indices are drawn per
    class WITH replacement — minority classes are orders of magnitude smaller
    than the balanced stream consumes (rot_ccw ~0.1% of frames), so each of
    their frames recurs many times while the majority class is subsampled; this
    is the intended trade. Loss is UNWEIGHTED cross-entropy — the balance lives
    in the sampler, not the loss (no residual weights). Yields 3-tuples
    ``(stacks, labels, aux_targets[B,2])``."""
    actions = np.asarray(dataset.actions)
    targets = np.asarray(dataset.targets)
    pools = [np.flatnonzero(actions == a) for a in range(NUM_ACTIONS)]
    present = [p for p in pools if len(p)]
    k = len(present)
    base, rem = divmod(batch_size, k)
    for _ in range(total_steps):
        parts = [
            pool[rng.integers(0, len(pool), size=base + (1 if j < rem else 0))]
            for j, pool in enumerate(present)
        ]
        idx = np.concatenate(parts)
        rng.shuffle(idx)
        yield dataset.batch_stacks(idx), actions[idx].astype(np.int64), targets[idx]


def noop_press_batches(dataset, batch_size: int, total_steps: int,
                       rng: np.random.Generator):
    """50/50 noop/press batch stream (PLAN2.md §6 covariate-shift amendment).

    Yields exactly ``total_steps`` ``(stacks[B,4,96,96] f32, labels[B] int64)``
    batches: half of each batch is drawn from the noop pool, the other half
    uniformly across the PRESS classes present in ``dataset`` (remainder spread
    one-per-class). Softer than the 5-way-uniform :func:`balanced_batches` —
    that inflated press priors ~4x over noop and caused false-press thrashing
    in closed loop. Per-class draws are WITH replacement; loss stays UNWEIGHTED
    cross-entropy (the balance lives in the sampler). Yields 3-tuples
    ``(stacks, labels, aux_targets[B,2])``."""
    actions = np.asarray(dataset.actions)
    targets = np.asarray(dataset.targets)
    noop_pool = np.flatnonzero(actions == 0)
    press_pools = [p for p in (np.flatnonzero(actions == a)
                               for a in range(1, NUM_ACTIONS)) if len(p)]
    if not len(noop_pool) or not press_pools:
        # Degenerate corpus (single side only): fall back to uniform-over-present.
        yield from balanced_batches(dataset, batch_size, total_steps, rng)
        return
    n_noop = batch_size // 2
    k = len(press_pools)
    base, rem = divmod(batch_size - n_noop, k)
    for _ in range(total_steps):
        parts = [noop_pool[rng.integers(0, len(noop_pool), size=n_noop)]]
        for j, pool in enumerate(press_pools):
            parts.append(pool[rng.integers(0, len(pool),
                                           size=base + (1 if j < rem else 0))])
        idx = np.concatenate(parts)
        rng.shuffle(idx)
        yield dataset.batch_stacks(idx), actions[idx].astype(np.int64), targets[idx]


# §6 perception amendment: aux (rot, col) CE weight. Set 0.1 (not the original
# 0.5): the attempt-5 mini ablation showed the dense 14-way aux gradients crowd
# out the sparse action signal at 0.5 (off-manifold press recall 0.27x vs 1.28x
# without aux at an 8k-step budget).
AUX_LOSS_WEIGHT = 0.1


def train_step(model, optimizer, stacks: np.ndarray, labels: np.ndarray,
               device: str, targets: np.ndarray | None = None) -> float:
    """One optimizer step: unweighted action cross-entropy plus, where aux
    targets are defined (rows != AUX_IGNORE), ``AUX_LOSS_WEIGHT * (CE(rot) +
    CE(col))`` on the aux heads (§6 perception amendment; train-time only —
    inference is still argmax over the action head). Returns the scalar total
    loss. (Class balance is the sampler's job.)"""
    import torch
    import torch.nn.functional as F

    model.train()
    x = torch.from_numpy(stacks).to(device)
    y = torch.from_numpy(labels).to(device)
    logits, _, (aux_rot, aux_col) = model(x, return_aux=True)
    loss = F.cross_entropy(logits, y)
    if targets is not None:
        t = torch.from_numpy(targets.astype(np.int64)).to(device)
        mask = t[:, 0] != AUX_IGNORE
        if bool(mask.any()):
            loss = loss + AUX_LOSS_WEIGHT * (
                F.cross_entropy(aux_rot[mask], t[mask, 0])
                + F.cross_entropy(aux_col[mask], t[mask, 1])
            )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def per_class_accuracy(model, dataset, device: str, n_samples: int,
                       rng: np.random.Generator, chunk: int = 512) -> dict:
    """Held-in per-class argmax accuracy over ``n_samples`` random frames. Not a
    metrics-schema field — stored in checkpoint metadata / printed only."""
    import torch

    n = len(dataset)
    n_samples = min(n_samples, n)
    idx = rng.choice(n, size=n_samples, replace=False)
    labels = dataset.actions[idx].astype(np.int64)
    preds = np.empty(n_samples, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for s in range(0, n_samples, chunk):
            sl = slice(s, s + chunk)
            x = torch.from_numpy(dataset.batch_stacks(idx[sl])).to(device)
            logits, _ = model(x)
            preds[sl] = logits.argmax(dim=1).cpu().numpy()
    out = {}
    for a in range(NUM_ACTIONS):
        m = labels == a
        tot = int(m.sum())
        out[ACTIONS[a]] = (float((preds[m] == a).mean()) if tot else None)
    out["overall"] = float((preds == labels).mean())
    return out


# --------------------------------------------------------------------------
# Closed-loop policy eval (SHARED with Phase E PPO) — vectorized over games
# --------------------------------------------------------------------------


class PolicyEval:
    __slots__ = ("lines", "pieces", "seeds", "moves", "median_lines",
                 "mean_lines", "p10_lines", "mean_pieces", "best_index")

    def __init__(self, lines, pieces, seeds, moves):
        self.lines = lines
        self.pieces = pieces
        self.seeds = seeds
        self.moves = moves
        self.median_lines = float(statistics.median(lines)) if lines else 0.0
        self.mean_lines = float(statistics.mean(lines)) if lines else 0.0
        self.p10_lines = float(p10(lines)) if lines else 0.0
        self.mean_pieces = float(statistics.mean(pieces)) if pieces else 0.0
        self.best_index = int(np.argmax(lines)) if lines else 0


def _greedy_actions(model, batch: np.ndarray, device: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        logits, _ = model(torch.from_numpy(batch).to(device))
        return logits.argmax(dim=1).cpu().numpy()


# Opposing-action pairs for the thrash metric (a press "reversed" by its opposite
# within the next 2 decisions): LEFT<->RIGHT, ROT_CW<->ROT_CCW.
from tetris.frame_env import LEFT, NOOP, RIGHT, ROT_CCW, ROT_CW  # noqa: E402

_REVERSAL = {LEFT: RIGHT, RIGHT: LEFT, ROT_CW: ROT_CCW, ROT_CCW: ROT_CW}


def _biased_actions(model, batch: np.ndarray, device: str,
                    logit_bias: np.ndarray | None) -> np.ndarray:
    """Greedy argmax over logits, optionally adding a per-class ``logit_bias``
    (shape ``(5,)``) BEFORE the argmax — the inference-time log-prior calibration
    (balanced-sampler training learns p_bal(a|s); adding scale*log(natural prior)
    recovers a natural-posterior-like decision boundary and cuts false presses)."""
    import torch

    with torch.no_grad():
        logits, _ = model(torch.from_numpy(batch).to(device))
        logits = logits.detach().cpu().numpy()
    if logit_bias is not None:
        logits = logits + np.asarray(logit_bias, dtype=np.float64).reshape(1, -1)
    return logits.argmax(axis=1)


def thrash_score(actions: list[int]) -> tuple[int, int]:
    """(reversed_presses, total_presses) for one game's decision sequence.

    A decision is a PRESS if it is not ``noop``. A press at decision ``i`` is
    *reversed* if either of decisions ``i+1`` or ``i+2`` is its opposing action
    (LEFT<->RIGHT, ROT_CW<->ROT_CCW) — an immediate oscillation the expert never
    makes. The thrash score is ``reversed_presses / total_presses`` (0 if no
    presses). Documented precisely so the demo/report number is reproducible."""
    presses = reversed_ = 0
    n = len(actions)
    for i, a in enumerate(actions):
        if a == NOOP:
            continue
        presses += 1
        opp = _REVERSAL[a]
        if (i + 1 < n and actions[i + 1] == opp) or (i + 2 < n and actions[i + 2] == opp):
            reversed_ += 1
    return reversed_, presses


def evaluate_policy_rich(model, seeds, max_pieces: int, device: str = "cpu",
                         logit_bias: np.ndarray | None = None) -> dict:
    """Closed-loop greedy eval (same lockstep machinery as :func:`evaluate_policy`)
    with the toddler-spam diagnostics: per-action histogram, presses-per-piece,
    press fraction, and the :func:`thrash_score`. ``logit_bias`` (optional ``(5,)``)
    is added to the logits before argmax (inference-time log-prior calibration).

    Returns a dict of aggregate stats over the games plus raw per-game lines/pieces.
    """
    seeds = [int(s) for s in seeds]
    n = len(seeds)
    envs = [FrameEnv(seed=s) for s in seeds]
    hist = [deque(maxlen=4) for _ in range(n)]
    act_seq: list[list[int]] = [[] for _ in range(n)]
    done = [e.game_over or e.pieces >= max_pieces for e in envs]
    model.eval()

    while not all(done):
        active = [i for i in range(n) if not done[i]]
        if envs[active[0]].is_decision_tick:
            batch = np.empty((len(active), 4, OBS_SIZE, OBS_SIZE), dtype=np.float32)
            for bi, i in enumerate(active):
                obs = render_env(envs[i])
                h = hist[i]
                if not h:
                    for _ in range(4):
                        h.append(obs)
                else:
                    h.append(obs)
                batch[bi] = np.stack(h).astype(np.float32) / 255.0
            acts = _biased_actions(model, batch, device, logit_bias)
            for bi, i in enumerate(active):
                a = int(acts[bi])
                envs[i].apply_action(a)
                act_seq[i].append(a)
        for i in active:
            envs[i].tick()
            if envs[i].game_over or envs[i].pieces >= max_pieces:
                done[i] = True

    lines = [e.lines for e in envs]
    pieces = [e.pieces for e in envs]
    class_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    total_decisions = 0
    rev_total = press_total = 0
    for seq in act_seq:
        for a in seq:
            class_counts[a] += 1
        total_decisions += len(seq)
        r, p = thrash_score(seq)
        rev_total += r
        press_total += p
    total_pieces = int(sum(pieces))
    return {
        "median_lines": float(statistics.median(lines)) if lines else 0.0,
        "mean_lines": float(statistics.mean(lines)) if lines else 0.0,
        "p10_lines": float(p10(lines)) if lines else 0.0,
        "pieces_per_game": float(statistics.mean(pieces)) if pieces else 0.0,
        "presses_per_piece": (press_total / total_pieces) if total_pieces else 0.0,
        "press_frac": (press_total / total_decisions) if total_decisions else 0.0,
        "thrash": (rev_total / press_total) if press_total else 0.0,
        "action_hist": {ACTIONS[a]: (int(class_counts[a]),
                                     float(class_counts[a] / total_decisions)
                                     if total_decisions else 0.0)
                        for a in range(NUM_ACTIONS)},
        "n_games": n,
        "total_pieces": total_pieces,
        "total_presses": int(press_total),
        "lines": [int(x) for x in lines],
        "pieces": [int(x) for x in pieces],
    }


def evaluate_policy(model, seeds, max_pieces: int, device: str = "cpu") -> PolicyEval:
    """Closed-loop greedy real-time eval over ``seeds`` (headless ticks, argmax
    over logits at decision ticks). Games run in lockstep so their decision-tick
    forwards batch into one call. Reused by BC, DAgger, and PPO evals."""
    seeds = [int(s) for s in seeds]
    n = len(seeds)
    envs = [FrameEnv(seed=s) for s in seeds]
    hist = [deque(maxlen=4) for _ in range(n)]
    moves: list[list[list[int]]] = [[] for _ in range(n)]
    done = [e.game_over or e.pieces >= max_pieces for e in envs]
    model.eval()

    while not all(done):
        active = [i for i in range(n) if not done[i]]
        if envs[active[0]].is_decision_tick:
            batch = np.empty((len(active), 4, OBS_SIZE, OBS_SIZE), dtype=np.float32)
            for bi, i in enumerate(active):
                obs = render_env(envs[i])
                h = hist[i]
                if not h:
                    for _ in range(4):
                        h.append(obs)
                else:
                    h.append(obs)
                batch[bi] = np.stack(h).astype(np.float32) / 255.0
            acts = _greedy_actions(model, batch, device)
            for bi, i in enumerate(active):
                envs[i].apply_action(int(acts[bi]))
        for i in active:
            lock = envs[i].tick()
            if lock is not None:
                moves[i].append([lock["r"], lock["c"]])
            if envs[i].game_over or envs[i].pieces >= max_pieces:
                done[i] = True

    return PolicyEval([e.lines for e in envs], [e.pieces for e in envs], seeds, moves)


# --------------------------------------------------------------------------
# DAgger — student rollout + expert relabel of every visited decision frame
# --------------------------------------------------------------------------


def dagger_rollout(model, teacher, target_frames: int, base_seed: int,
                   device: str = "cpu", max_pieces: int = 10000,
                   progress: bool = False) -> dict:
    """Roll out the STUDENT (greedy argmax) on fresh seeds for ~``target_frames``
    decision frames; relabel EVERY visited decision frame with the expert's
    action for THAT state (current-pose replan, :class:`DaggerRelabeler`), which
    a spawn-time script cannot do once the piece has moved. Returns packed
    frames + relabels + an episodes table for :func:`write_dagger_dataset`."""
    import torch  # noqa: F401  (device tensors created in _greedy_actions)

    packed: list[np.ndarray] = []
    actions = array("B")
    ep_ids = array("i")
    targets = array("B")  # flat (rot, col) aux pairs, AUX_IGNORE when undefined
    episodes = []
    total = 0
    ep = 0
    import time
    t0 = time.perf_counter()
    while total < target_frames:
        seed = base_seed + ep
        env = FrameEnv(seed=seed)
        relab = DaggerRelabeler(teacher)
        h: deque = deque(maxlen=4)
        start = total
        while (not env.game_over and env.pieces < max_pieces
               and total < target_frames):
            if env.is_decision_tick:
                obs = render_env(env)  # the state the student saw
                if not h:
                    for _ in range(4):
                        h.append(obs)
                else:
                    h.append(obs)
                stack = (np.stack(h).astype(np.float32) / 255.0)[None]
                student_a = int(_greedy_actions(model, stack, device)[0])
                label, t_rot, t_col = relab.relabel_with_target(env)
                packed.append(pack_obs(obs))
                actions.append(label)
                ep_ids.append(ep)
                targets.append(t_rot)
                targets.append(t_col)
                total += 1
                env.apply_action(student_a)
            env.tick()
        episodes.append((start, total - start, seed, int(env.lines), int(env.pieces)))
        ep += 1
        if progress:
            el = time.perf_counter() - t0
            print(f"  dagger ep {ep:>3} seed={seed} lines={env.lines:>5} "
                  f"pieces={env.pieces:>5} | frames {total:>7}/{target_frames} "
                  f"{total / max(el, 1e-9):.0f}/s", flush=True)
    return {
        "packed": packed,
        "actions": np.frombuffer(actions, dtype=np.uint8).copy(),
        "episode_id": np.frombuffer(ep_ids, dtype=np.int32).copy(),
        "targets": np.frombuffer(targets, dtype=np.uint8).copy().reshape(-1, 2),
        "episodes": episodes,
        "n_frames": total,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


def write_dagger_dataset(out_dir: str | Path, roll: dict) -> Path:
    """Persist a :func:`dagger_rollout` result in the on-disk BCDataset format so
    it can be wrapped by :class:`BCDataset` and aggregated via
    :class:`MultiBCDataset`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_arr = (np.stack(roll["packed"]).astype(np.uint8)
                  if roll["packed"] else np.empty((0, PACKED_BYTES), np.uint8))
    frames_arr.tofile(out_dir / "frames.u8pack")
    np.save(out_dir / "actions.npy", roll["actions"])
    np.save(out_dir / "episode_id.npy", roll["episode_id"])
    if "targets" in roll:
        np.save(out_dir / "targets.npy", roll["targets"])
    hist = class_histogram(roll["actions"])
    n = int(roll["n_frames"])
    meta = {
        "n_frames": n,
        "n_episodes": len(roll["episodes"]),
        "obs_size": OBS_SIZE,
        "packed_bytes": PACKED_BYTES,
        "num_actions": NUM_ACTIONS,
        "actions": list(ACTIONS),
        "source": "dagger_rollout",
        "class_histogram": {ACTIONS[a]: int(hist[a]) for a in range(NUM_ACTIONS)},
        "episodes": roll["episodes"],
        "elapsed_s": roll.get("elapsed_s"),
        "files": {"frames": "frames.u8pack", "actions": "actions.npy",
                  "episode_id": "episode_id.npy"},
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Generate the BC dataset (PLAN2.md §5/§6)")
    ap.add_argument("--out", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--pieces", type=int, default=25000, help="target total pieces")
    ap.add_argument("--max-game-pieces", type=int, default=1000)
    ap.add_argument("--base-seed", type=int, default=100000)
    ap.add_argument("--teacher", choices=["td", "cem"], default="td")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dart", action="store_true",
                    help="DART noise-injected generation (§6 amendment)")
    ap.add_argument("--noise-seed", type=int, default=7)
    args = ap.parse_args(argv)

    if args.dart:
        generate_dart_dataset(
            out_dir=args.out,
            total_pieces=args.pieces,
            max_game_pieces=args.max_game_pieces,
            base_seed=args.base_seed,
            noise_seed=args.noise_seed,
            teacher_kind=args.teacher,
            checkpoint=args.checkpoint,
            device=args.device,
        )
        return 0

    generate_dataset(
        out_dir=args.out,
        total_pieces=args.pieces,
        max_game_pieces=args.max_game_pieces,
        base_seed=args.base_seed,
        teacher_kind=args.teacher,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
