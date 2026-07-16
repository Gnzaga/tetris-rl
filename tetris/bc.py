"""Behavioral-cloning dataset generation + class weighting (PLAN2.md §5, Phase C).

This module currently holds the Phase C **dataset half** only: generating the
keypress-expert demonstration corpus, packing observations to disk, and the
class histogram / inverse-frequency weights. The BC + DAgger **training loops**
are Phase D and will be added to this same file (per the PLAN2.md §2 layout).

Storage format (runs/bc_data_v1/ by default; runs/ is gitignored)
------------------------------------------------------------------
The expert plays seeded frame-layer games; at every decision tick we store the
96x96 observation the agent saw (rendered BEFORE its action) plus the action it
chose. Observations are strictly binary (0 / 255), so each frame is bit-packed
with :func:`numpy.packbits` to 1152 bytes (96*96 / 8) — an 8x saving over raw
uint8 (which would be ~35 GB for ~3.8M frames). Files:

* ``frames.u8pack`` : raw ``(N, 1152)`` uint8 memmap of packed frames.
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
PACKED_BYTES = OBS_SIZE * OBS_SIZE // 8  # 1152
WEIGHT_CAP = 20.0  # inverse-frequency weights clipped to this multiple of noop


# --------------------------------------------------------------------------
# Obs <-> packed-bits helpers
# --------------------------------------------------------------------------


def pack_obs(obs: np.ndarray) -> np.ndarray:
    """(96,96) uint8 {0,255} -> (1152,) uint8 packed bits (numpy MSB-first)."""
    return np.packbits(obs.reshape(-1) != 0)


def unpack_obs(packed: np.ndarray) -> np.ndarray:
    """(1152,) packed bits -> (96,96) uint8 {0,255} (inverse of :func:`pack_obs`)."""
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))
    return (bits.astype(np.uint8) * 255).reshape(OBS_SIZE, OBS_SIZE)


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
        "packing": "np.packbits(obs.reshape(-1) != 0), MSB-first, 1152 bytes/frame",
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
# Dataset reader (used by Phase C tests and, later, Phase D training)
# --------------------------------------------------------------------------


class BCDataset:
    """Random-access reader over a generated dataset. Reconstructs normalized
    4-stacks by index (never crossing an episode boundary)."""

    def __init__(self, out_dir: str | Path = DEFAULT_DATA_DIR):
        out_dir = Path(out_dir)
        with open(out_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.n = int(self.meta["n_frames"])
        self.frames = np.memmap(
            out_dir / "frames.u8pack", dtype=np.uint8, mode="r",
            shape=(self.n, PACKED_BYTES),
        )
        self.actions = np.load(out_dir / "actions.npy")
        self.episode_id = np.load(out_dir / "episode_id.npy")
        # Per-frame episode-start index for O(1) stack reconstruction.
        self.ep_start = np.empty(self.n, dtype=np.int64)
        for start, length, *_ in self.meta["episodes"]:
            self.ep_start[start:start + length] = start

    def __len__(self) -> int:
        return self.n

    def frame(self, i: int) -> np.ndarray:
        """(96,96) uint8 {0,255} observation at index ``i``."""
        return unpack_obs(self.frames[i])

    def stack_indices(self, i: int) -> list[int]:
        s = int(self.ep_start[i])
        return [max(i - 3, s), max(i - 2, s), max(i - 1, s), i]

    def stack(self, i: int) -> np.ndarray:
        """(4,96,96) float32 in [0,1]: the last-4-observation policy input."""
        idx = self.stack_indices(i)
        bits = np.unpackbits(self.frames[idx], axis=1)  # (4, 9216)
        return bits.astype(np.float32).reshape(4, OBS_SIZE, OBS_SIZE)

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
        bits = np.unpackbits(self.frames[flat], axis=1)  # (B*4, 9216)
        return bits.astype(np.float32).reshape(len(idx), 4, OBS_SIZE, OBS_SIZE)


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
    return epochs * math.ceil(n / batch_size)


def iter_minibatches(dataset, batch_size: int, epochs: int, rng: np.random.Generator):
    """Yield ``(epoch, stacks[B,4,96,96] f32, labels[B] int64)`` for ``epochs``
    reshuffled passes over ``dataset`` (drop no tail; last batch may be short)."""
    n = len(dataset)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            yield ep, dataset.batch_stacks(idx), dataset.actions[idx].astype(np.int64)


def train_step(model, optimizer, stacks: np.ndarray, labels: np.ndarray,
               weight, device: str) -> float:
    """One class-weighted cross-entropy optimizer step; returns the scalar loss."""
    import torch
    import torch.nn.functional as F

    model.train()
    x = torch.from_numpy(stacks).to(device)
    y = torch.from_numpy(labels).to(device)
    logits, _ = model(x)
    loss = F.cross_entropy(logits, y, weight=weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def weight_tensor(weights: np.ndarray, device: str):
    import torch

    return torch.tensor(np.asarray(weights, dtype=np.float32), device=device)


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
                label = relab.relabel(env)  # expert action for the CURRENT pose
                packed.append(pack_obs(obs))
                actions.append(label)
                ep_ids.append(ep)
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


def combined_class_weights(dataset) -> np.ndarray:
    """Capped inverse-frequency weights recomputed over a (possibly multi-shard)
    dataset's label distribution — DAgger shifts the class mix, so weights are
    refreshed each retrain rather than reused from the base meta."""
    return inverse_freq_weights(class_histogram(np.asarray(dataset.actions)))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Generate the BC dataset (PLAN2.md §5)")
    ap.add_argument("--out", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--pieces", type=int, default=25000, help="target total pieces")
    ap.add_argument("--max-game-pieces", type=int, default=1000)
    ap.add_argument("--base-seed", type=int, default=100000)
    ap.add_argument("--teacher", choices=["td", "cem"], default="td")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

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
