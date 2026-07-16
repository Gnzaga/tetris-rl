"""TD(0) afterstate trainer, replay buffer, and schedules (PLAN.md §8).

Synchronous vectorized self-play in one process: ``num_envs`` parallel games,
every step all envs' candidate afterstates are concatenated into ONE forward
pass through the online ValueNet. Transitions ``(A_prev, r, A_curr, done)`` land
in a replay buffer that stores boards compactly as 20 ``uint16`` rows and
decodes them to tensors only on sample. The TD(0) afterstate target is

    V(A_prev) <- r + gamma * V_bar(A_curr) * (1 - done)

with a target network ``V_bar`` hard-copied every ``target_sync`` updates and no
bootstrap on terminal transitions (whose reward already carries the -10 death
penalty). Huber loss, Adam, roughly one gradient step per vector-env step.

``scripts/train_td.py`` owns the run directory, CLI, rich table, periodic eval,
and checkpointing; this module is the algorithm + buffer + schedules so it stays
unit-testable with tiny nets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .agents import LinearAgent, decision_rewards
from .engine import TetrisEngine
from .evaluation import EvalResult
from .model import boards_to_tensor

TERMINAL_PENALTY = -10.0


# --- schedules (PLAN.md §2, §8) ---------------------------------------------


def epsilon_at(
    pieces: int, budget: int, eps_start: float, eps_end: float = 0.02, frac: float = 0.30
) -> float:
    """Linear epsilon from ``eps_start`` to ``eps_end`` over the first ``frac`` of
    the budget, then flat at ``eps_end`` (PLAN.md §8: 1.0->0.02 over first 30%;
    warm-start starts at 0.2)."""
    span = frac * budget
    if span <= 0 or pieces >= span:
        return eps_end
    return eps_start + (eps_end - eps_start) * (pieces / span)


def beta_at(pieces: int, budget: int, frac: float = 0.60) -> float:
    """Linear shaping weight 1 -> 0 over the first ``frac`` of the budget, then 0
    (PLAN.md §2: beta anneals over the first 60%)."""
    span = frac * budget
    if span <= 0 or pieces >= span:
        return 0.0
    return 1.0 - pieces / span


def td_target(
    rewards: np.ndarray, next_values: np.ndarray, dones: np.ndarray, gamma: float
) -> np.ndarray:
    """TD(0) afterstate target: ``r + gamma * V_bar(A') * (1 - done)``.

    Terminal transitions (``done``) do not bootstrap; their reward already
    includes the -10 death penalty.
    """
    return rewards + gamma * next_values * (1.0 - np.asarray(dones, dtype=np.float32))


# --- replay buffer ----------------------------------------------------------


class ReplayBuffer:
    """Ring buffer of afterstate transitions; boards stored as 20 uint16 rows."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.states = np.zeros((self.capacity, 20), dtype=np.uint16)
        self.next_states = np.zeros((self.capacity, 20), dtype=np.uint16)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def add_many(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        n = len(states)
        if n == 0:
            return
        states = np.asarray(states, dtype=np.uint16).reshape(n, 20)
        next_states = np.asarray(next_states, dtype=np.uint16).reshape(n, 20)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(n)
        dones = np.asarray(dones, dtype=np.float32).reshape(n)
        # Write with wraparound.
        idx = (self.pos + np.arange(n)) % self.capacity
        self.states[idx] = states
        self.next_states[idx] = next_states
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.pos = int((self.pos + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        """Return ``(states, rewards, next_states, dones)`` numpy arrays."""
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


# --- vectorized self-play ---------------------------------------------------


class VecSelfPlay:
    """``num_envs`` parallel games producing afterstate transitions.

    Holds each env's previous afterstate ``A_prev`` so a transition can be emitted
    once the next placement's reward and afterstate are known. When a game ends it
    is reset to a fresh deterministic seed drawn from an internal counter.
    """

    def __init__(self, num_envs: int, base_seed: int):
        self.num_envs = num_envs
        self._next_seed = base_seed + num_envs
        self.engines = [TetrisEngine(seed=base_seed + i) for i in range(num_envs)]
        self.prev: list[np.ndarray | None] = [None] * num_envs

    def _reset(self, i: int) -> None:
        self.engines[i] = TetrisEngine(seed=self._next_seed)
        self._next_seed += 1
        self.prev[i] = None

    def step(
        self, model, beta: float, epsilon: float, gamma: float, rng, device: str
    ):
        """Advance every active env by one placement.

        Returns ``(states, rewards, next_states, dones, num_stepped)`` — the
        transitions produced this step (envs whose game had no predecessor
        afterstate yet contribute none). One forward pass scores all candidates.
        """
        cand = []  # (env_idx, placements, afters, r)
        for i, e in enumerate(self.engines):
            res = decision_rewards(e, beta)
            if res is None:  # unreachable for a non-terminal engine, guard anyway
                self._reset(i)
                continue
            placements, afters, r = res
            cand.append((i, placements, afters, r))

        if not cand:
            return (
                np.empty((0, 20), np.uint16),
                np.empty(0, np.float32),
                np.empty((0, 20), np.uint16),
                np.empty(0, np.float32),
                0,
            )

        # Single batched forward over every candidate afterstate of every env.
        all_afters = np.concatenate([c[2] for c in cand], axis=0)
        model.eval()
        with torch.no_grad():
            values = (
                model(boards_to_tensor(all_afters, device)).detach().cpu().numpy().reshape(-1)
            )

        s_out, r_out, ns_out, d_out = [], [], [], []
        num_stepped = 0
        off = 0
        for i, placements, afters, r in cand:
            p = len(placements)
            v = values[off : off + p]
            off += p
            q = r + gamma * v

            if epsilon > 0.0 and rng is not None and rng.next_float() < epsilon:
                j = int(rng.next_float() * p)
                if j >= p:
                    j = p - 1
            else:
                j = int(np.argmax(q))

            e = self.engines[i]
            info = e.step(*placements[j])
            num_stepped += 1
            done = bool(info.game_over)
            a_curr = afters[j]
            reward = float(r[j]) + (TERMINAL_PENALTY if done else 0.0)

            if self.prev[i] is not None:
                s_out.append(self.prev[i])
                r_out.append(reward)
                ns_out.append(a_curr)
                d_out.append(1.0 if done else 0.0)

            if done:
                self._reset(i)
            else:
                self.prev[i] = a_curr

        return (
            np.asarray(s_out, dtype=np.uint16).reshape(-1, 20),
            np.asarray(r_out, dtype=np.float32),
            np.asarray(ns_out, dtype=np.uint16).reshape(-1, 20),
            np.asarray(d_out, dtype=np.float32),
            num_stepped,
        )


# --- greedy vectorized evaluation -------------------------------------------


def evaluate_net(
    model,
    seeds,
    max_pieces: int,
    gamma: float,
    device: str = "cpu",
    record: bool = False,
) -> EvalResult:
    """Play ``len(seeds)`` greedy games in parallel with the ValueNet (beta=0).

    All active games' candidates share one forward pass per step. Deterministic
    given the model + seeds. When ``record`` is set, the best game's move-list is
    available via ``EvalResult.moves[best_index]``.
    """
    seeds = list(seeds)
    n = len(seeds)
    engines = [TetrisEngine(seed=s) for s in seeds]
    lines = [0] * n
    pieces = [0] * n
    moves: list[list[list[int]] | None] = [[] if record else None for _ in range(n)]

    def _active(i: int) -> bool:
        return not engines[i].game_over and engines[i].pieces < max_pieces

    model.eval()
    while any(_active(i) for i in range(n)):
        cand = []
        for i in range(n):
            if not _active(i):
                continue
            res = decision_rewards(engines[i], 0.0)
            if res is None:
                continue
            cand.append((i, *res))
        if not cand:
            break

        all_afters = np.concatenate([c[2] for c in cand], axis=0)
        with torch.no_grad():
            values = (
                model(boards_to_tensor(all_afters, device)).detach().cpu().numpy().reshape(-1)
            )

        off = 0
        for i, placements, afters, r in cand:
            p = len(placements)
            v = values[off : off + p]
            off += p
            j = int(np.argmax(r + gamma * v))
            move = placements[j]
            if moves[i] is not None:
                moves[i].append([int(move[0]), int(move[1])])
            engines[i].step(*move)
            lines[i] = engines[i].lines
            pieces[i] = engines[i].pieces

    return EvalResult(list(seeds), lines, pieces, moves)


# --- warm-start (behavior cloning of the CEM teacher, PLAN.md §8) -----------


@dataclass
class TeacherData:
    boards: np.ndarray  # (M, 20) uint16 — every candidate afterstate seen
    targets: np.ndarray  # (M,) float32 — per-decision z-normalized teacher scores


def collect_teacher_data(
    teacher_weights, num_placements: int, base_seed: int
) -> TeacherData:
    """Roll the linear teacher out for ``num_placements`` decisions, recording
    every candidate afterstate and the teacher's linear scores normalized to zero
    mean / unit variance *within each decision* (PLAN.md §8)."""
    weights = np.asarray(teacher_weights, dtype=np.float64)
    agent = LinearAgent(weights)
    boards_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []

    count = 0
    seed = base_seed
    while count < num_placements:
        engine = TetrisEngine(seed=seed)
        seed += 1
        while not engine.game_over and count < num_placements:
            placements, feats, afters = engine.candidate_features()
            if not placements:
                break
            scores = feats @ weights
            mu = scores.mean()
            sd = scores.std()
            tgt = (scores - mu) / sd if sd > 1e-8 else np.zeros_like(scores)
            boards_chunks.append(np.asarray(afters, dtype=np.uint16))
            target_chunks.append(tgt.astype(np.float32))
            move = agent.act(engine)
            engine.step(*move)
            count += 1

    boards = np.concatenate(boards_chunks, axis=0)
    targets = np.concatenate(target_chunks, axis=0)
    return TeacherData(boards, targets)


def pretrain_regression(
    model,
    data: TeacherData,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int = 0,
) -> list[float]:
    """MSE-regress V onto per-decision normalized teacher scores. Returns the mean
    loss of each epoch (monotone-ish decreasing on a well-posed regression)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(data.boards)
    model.train()
    epoch_losses: list[float] = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        nb = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x = boards_to_tensor(data.boards[idx], device)
            y = torch.from_numpy(data.targets[idx]).to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        epoch_losses.append(total / max(nb, 1))
    return epoch_losses
