"""Minimal PPO — the pure-RL comparison arm (PLAN2.md §7, Phase E).

This is the *honest-contrast* trainer: the exact same :class:`tetris.policy_model.PolicyNet`
(pi + value heads) as BC, but trained **from scratch** by PPO-clip on the frame
layer's sparse reward. No BC init, no shaping — frame-level pixel Tetris is the
regime where classical RL is expected to fail, and the point of this arm is to
measure that failure honestly against the BC/DAgger agent (there is NO
performance gate).

Frozen hyperparameters (PLAN2.md §7)
-----------------------------------
* PPO-clip, clip epsilon ``0.2`` (standard).
* GAE with ``gamma = 0.99``, ``lambda = 0.95``.
* Entropy bonus ``0.01``; value-loss coefficient ``0.5`` (standard).
* ``16`` parallel frame envs, rollout ``128`` decisions/env (2048 samples/batch).
* ``4`` optimizer epochs per batch, Adam ``lr = 2.5e-4``.

Reward (PLAN2.md §1, no shaping): at each *decision* the reward is
``clear_points[lines]`` for any line-clearing lock that occurred in the tick
window immediately following that decision's action, plus a ``-10`` terminal on
game over. Nothing else.

Observations reuse the BC stacking convention (``tetris.bc`` /
``evaluate_policy``): a 4-stack of the last decision-tick 96x96 frames,
normalized to [0,1]; an episode's first frames repeat its first frame.

Time-box (hard, PLAN2.md §7): the driver in ``scripts/train_ppo.py`` runs until
``--max-hours`` OR ``--max-frames`` (decision frames), whichever comes first,
then writes a final checkpoint + final eval regardless of performance. SIGTERM is
handled the same way (checkpoint + clean exit). Everything here is torch-only, no
new deps.

Losses / metrics semantics (documented for the frozen ``tetris.runio`` schema):
the ``loss`` logged to metrics.jsonl is the **total** PPO loss
(``policy + vf_coef*value - ent_coef*entropy``); ``pieces_trained`` carries the
**decision frames consumed** (``n_envs * rollout_len`` accumulated) — the PPO
x-axis. ``epsilon`` / ``beta`` are null (not applicable to PPO).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from tetris.engine import CLEAR_POINTS
from tetris.frame_env import DECISION_PERIOD, FrameEnv
from tetris.policy_model import OBS_SIZE
from tetris.render_obs import render_env

# Frozen PPO hyperparameters (PLAN2.md §7).
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
LR = 2.5e-4
N_ENVS = 16
ROLLOUT_LEN = 128
EPOCHS = 4

TERMINAL_REWARD = -10.0  # PLAN2.md §1


# ==========================================================================
# Per-decision environment stepping (reward placement, PLAN2.md §1)
# ==========================================================================


def step_decision(env: FrameEnv, action: int) -> tuple[float, bool]:
    """Apply ``action`` at the current decision tick, then advance
    ``DECISION_PERIOD`` ticks to the next decision tick, accumulating reward.

    The reward for this decision is ``clear_points[lines]`` for every
    line-clearing lock that fell in this tick window (PLAN2.md §1: the reward on
    the decision at/after the lock tick), plus ``-10`` on game over. ``env`` must
    be on a decision tick. Returns ``(reward, done)``; when ``done`` the caller is
    responsible for resetting ``env``.
    """
    env.apply_action(action)
    reward = 0.0
    for _ in range(DECISION_PERIOD):
        lines_before = env.lines
        lock = env.tick()
        if lock is not None:
            reward += CLEAR_POINTS[env.lines - lines_before]
        if env.game_over:
            return reward + TERMINAL_REWARD, True
    return reward, False


class VecFrameEnv:
    """A batch of ``n`` frame envs stepped in lockstep, with autoreset and the
    BC 4-stack observation convention.

    All envs start on a decision tick (``tick_count == 0``) and advance exactly
    one decision per :meth:`step`, so they stay phase-aligned; a terminated env is
    reset in place with a fresh, deterministically-allocated seed and its 4-frame
    history cleared. :meth:`observe` is idempotent between steps (calling it twice
    without an intervening :meth:`step` returns the same stacks and does not
    double-push history) so the rollout's bootstrap observation and the next
    rollout's first observation coincide.
    """

    def __init__(self, n: int = N_ENVS, base_seed: int = 0):
        self.n = n
        self.envs = [FrameEnv(seed=base_seed + i) for i in range(n)]
        self._next_seed = base_seed + n
        self._hist: list[list[np.ndarray]] = [[] for _ in range(n)]
        self._fresh = [True] * n  # a new decision-tick obs is available to push

    def observe(self) -> np.ndarray:
        """``(n, 4, 96, 96)`` float32 in [0,1]: the current stacked observation of
        every env (idempotent between steps)."""
        out = np.empty((self.n, 4, OBS_SIZE, OBS_SIZE), dtype=np.float32)
        for i, env in enumerate(self.envs):
            h = self._hist[i]
            if self._fresh[i] or not h:
                obs = render_env(env)
                if not h:
                    h.extend([obs, obs, obs, obs])  # cold start: repeat first frame
                else:
                    h.append(obs)
                    del h[:-4]
                self._fresh[i] = False
            out[i] = np.stack(h).astype(np.float32) / 255.0
        return out

    def step(self, actions) -> tuple[np.ndarray, np.ndarray]:
        """Apply one action per env, advance a decision, autoreset terminated
        envs. Returns ``(rewards[n] f32, dones[n] f32)`` for this decision."""
        rewards = np.empty(self.n, dtype=np.float32)
        dones = np.empty(self.n, dtype=np.float32)
        for i, env in enumerate(self.envs):
            r, done = step_decision(env, int(actions[i]))
            rewards[i] = r
            dones[i] = 1.0 if done else 0.0
            if done:
                env.reset(self._next_seed)
                self._next_seed += 1
                self._hist[i] = []
            self._fresh[i] = True  # a new decision-tick obs is available
        return rewards, dones


# ==========================================================================
# GAE + PPO math (pure functions, unit-tested against hand computation)
# ==========================================================================


def compute_gae(rewards, values, dones, last_values, gamma=GAMMA, lam=GAE_LAMBDA):
    """Generalized Advantage Estimation over a ``[T, N]`` rollout.

    ``rewards``/``values``/``dones`` are ``[T, N]``; ``dones[t]`` is 1.0 when the
    env terminated *during* step ``t`` (which masks bootstrapping past the episode
    boundary). ``last_values`` is ``[N]`` — the value of the observation after the
    final step, used to bootstrap step ``T-1``. Returns
    ``(advantages[T,N], returns[T,N])`` with ``returns = advantages + values``.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    last_values = np.asarray(last_values, dtype=np.float64)
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float64)
    gae = np.zeros(N, dtype=np.float64)
    next_value = last_values
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + values
    return adv.astype(np.float32), returns.astype(np.float32)


def normalize_advantages(adv, eps: float = 1e-8):
    """Zero-mean, unit-std advantage normalization over the flattened batch."""
    adv = np.asarray(adv, dtype=np.float32)
    return (adv - adv.mean()) / (adv.std() + eps)


def ppo_losses(logits, values, actions, old_logprobs, advantages, returns,
               clip=CLIP_EPS, vf_coef=VF_COEF, ent_coef=ENT_COEF):
    """PPO-clip losses for one minibatch (all torch tensors, batch-first).

    Returns a dict with ``policy`` (clipped surrogate, minimized), ``value``
    (``0.5*MSE(value, returns)``), ``entropy`` (mean policy entropy), ``total``
    (``policy + vf_coef*value - ent_coef*entropy``), and the diagnostics
    ``approx_kl`` and ``clip_frac``. ``advantages`` are used as given (normalize
    beforehand).
    """
    logp_all = F.log_softmax(logits, dim=-1)
    new_logp = logp_all.gather(1, actions.view(-1, 1)).squeeze(1)
    ratio = torch.exp(new_logp - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = 0.5 * (values - returns).pow(2).mean()
    entropy = -(logp_all.exp() * logp_all).sum(dim=-1).mean()
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy
    with torch.no_grad():
        approx_kl = (old_logprobs - new_logp).mean()
        clip_frac = ((ratio - 1.0).abs() > clip).float().mean()
    return {
        "policy": policy_loss,
        "value": value_loss,
        "entropy": entropy,
        "total": total,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }


# ==========================================================================
# Rollout collection + PPO update
# ==========================================================================


def _sample_actions(model, stacks_f32, device, gen):
    """Forward ``stacks`` [N,4,96,96] f32, sample one action per env from the
    policy (categorical), return ``(actions[N] int64, logprobs[N] f32,
    values[N] f32)`` as numpy. Sampling uses a CPU generator so it is
    device-independent and reproducible."""
    with torch.no_grad():
        x = torch.from_numpy(stacks_f32).to(device)
        logits, value = model(x)
        logp_all = F.log_softmax(logits, dim=-1).cpu()
        probs = logp_all.exp()
        actions = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        logp = logp_all.gather(1, actions.view(-1, 1)).squeeze(1)
    return (actions.numpy().astype(np.int64),
            logp.numpy().astype(np.float32),
            value.cpu().numpy().astype(np.float32))


def collect_rollout(model, vec: VecFrameEnv, rollout_len, device, gen):
    """Collect ``rollout_len`` decisions from every env in ``vec``.

    Returns a dict of flattened arrays over ``B = rollout_len * n_envs`` samples
    plus the GAE inputs. ``obs`` is stored as ``uint8`` {0,255}-scaled float→u8 to
    keep the buffer small; it is re-normalized in the update. Sampling is greedy-
    free (categorical) and reproducible given ``gen``."""
    N = vec.n
    T = rollout_len
    obs_buf = np.empty((T, N, 4, OBS_SIZE, OBS_SIZE), dtype=np.uint8)
    act_buf = np.empty((T, N), dtype=np.int64)
    logp_buf = np.empty((T, N), dtype=np.float32)
    val_buf = np.empty((T, N), dtype=np.float32)
    rew_buf = np.empty((T, N), dtype=np.float32)
    done_buf = np.empty((T, N), dtype=np.float32)

    model.eval()
    for t in range(T):
        stacks = vec.observe()  # (N,4,96,96) f32 in [0,1]
        obs_buf[t] = (stacks * 255.0 + 0.5).astype(np.uint8)
        actions, logp, value = _sample_actions(model, stacks, device, gen)
        act_buf[t] = actions
        logp_buf[t] = logp
        val_buf[t] = value
        rew_buf[t], done_buf[t] = vec.step(actions)

    # Bootstrap value of the post-rollout observation (idempotent observe).
    boot = vec.observe()
    with torch.no_grad():
        _, last_v = model(torch.from_numpy(boot).to(device))
    last_values = last_v.cpu().numpy().astype(np.float32)

    adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_values)
    B = T * N
    return {
        "obs": obs_buf.reshape(B, 4, OBS_SIZE, OBS_SIZE),
        "actions": act_buf.reshape(B),
        "logprobs": logp_buf.reshape(B),
        "values": val_buf.reshape(B),
        "advantages": adv.reshape(B),
        "returns": ret.reshape(B),
        "rewards": rew_buf,        # [T,N], kept for diagnostics
        "dones": done_buf,         # [T,N]
        "frames": B,
    }


def ppo_update(model, optimizer, batch, device, epochs=EPOCHS, minibatch_size=256,
               clip=CLIP_EPS, vf_coef=VF_COEF, ent_coef=ENT_COEF,
               max_grad_norm=0.5, rng=None):
    """Run ``epochs`` shuffled passes of PPO-clip updates over ``batch``.

    Advantages are normalized once over the full flattened batch (zero-mean/
    unit-std) before the epochs. Returns mean loss diagnostics over all minibatch
    steps (``total`` is the value logged to metrics.jsonl)."""
    if rng is None:
        rng = np.random.default_rng(0)
    B = batch["frames"]
    obs = batch["obs"]
    actions = torch.from_numpy(batch["actions"]).to(device)
    old_logp = torch.from_numpy(batch["logprobs"]).to(device)
    returns = torch.from_numpy(batch["returns"]).to(device)
    adv = torch.from_numpy(normalize_advantages(batch["advantages"])).to(device)

    model.train()
    sums = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0,
            "approx_kl": 0.0, "clip_frac": 0.0}
    steps = 0
    for _ep in range(epochs):
        perm = rng.permutation(B)
        for start in range(0, B, minibatch_size):
            mb = perm[start:start + minibatch_size]
            x = torch.from_numpy(obs[mb].astype(np.float32) / 255.0).to(device)
            logits, values = model(x)
            losses = ppo_losses(
                logits, values, actions[mb], old_logp[mb], adv[mb], returns[mb],
                clip=clip, vf_coef=vf_coef, ent_coef=ent_coef,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            for k in sums:
                sums[k] += float(losses[k].item())
            steps += 1
    return {k: v / max(steps, 1) for k, v in sums.items()}
