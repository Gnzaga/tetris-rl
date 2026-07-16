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
