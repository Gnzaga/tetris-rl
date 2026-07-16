"""PPO comparison-arm tests (PLAN2.md §7, Phase E).

Covers the load-bearing math and plumbing of :mod:`tetris.ppo` and the time-box
exit of ``scripts/train_ppo.py`` — WITHOUT running any long training:

* GAE against a hand-computed sequence, including terminal masking.
* PPO-clip loss math on synthetic logits/advantages (clip actually clips).
* Rollout buffer shapes + reward placement on the correct decision index
  (scripted env, one line clear).
* Time-box exit (patched clock) writes a final checkpoint and stops.
* Determinism of seeded rollout collection.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from rich.console import Console

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from tetris.engine import CLEAR_POINTS
from tetris.features import FULL_ROW
from tetris.frame_env import FrameEnv
from tetris.policy_model import NUM_ACTIONS, OBS_SIZE, PolicyNet
from tetris.ppo import (
    VecFrameEnv,
    collect_rollout,
    compute_gae,
    normalize_advantages,
    ppo_losses,
    step_decision,
)
from tetris.runio import RunWriter

import train_ppo  # noqa: E402


# --------------------------------------------------------------------------
# GAE — hand computation
# --------------------------------------------------------------------------

_GAMMA, _LAM = 0.99, 0.95


def test_gae_no_terminal_matches_hand_computation():
    rewards = np.array([[0.0], [1.0], [0.0]])
    values = np.array([[0.5], [0.4], [0.3]])
    dones = np.zeros((3, 1))
    last_values = np.array([0.2])
    adv, ret = compute_gae(rewards, values, dones, last_values, _GAMMA, _LAM)

    # Backward recursion by hand (gamma=0.99, lambda=0.95):
    d2 = 0.0 + _GAMMA * 0.2 - 0.3
    a2 = d2
    d1 = 1.0 + _GAMMA * 0.3 - 0.4
    a1 = d1 + _GAMMA * _LAM * a2
    d0 = 0.0 + _GAMMA * 0.4 - 0.5
    a0 = d0 + _GAMMA * _LAM * a1
    assert adv[:, 0] == pytest.approx([a0, a1, a2], rel=1e-6)
    assert ret[:, 0] == pytest.approx([a0 + 0.5, a1 + 0.4, a2 + 0.3], rel=1e-6)


def test_gae_terminal_masks_bootstrap():
    # Env terminates during step 1 => no bootstrap through that boundary and the
    # accumulated GAE resets there.
    rewards = np.array([[0.0], [1.0], [0.0]])
    values = np.array([[0.5], [0.4], [0.3]])
    dones = np.array([[0.0], [1.0], [0.0]])
    last_values = np.array([0.2])
    adv, _ = compute_gae(rewards, values, dones, last_values, _GAMMA, _LAM)

    a2 = 0.0 + _GAMMA * 0.2 - 0.3
    a1 = 1.0 - 0.4                       # nonterminal=0 kills bootstrap + prior gae
    a0 = (0.0 + _GAMMA * 0.4 - 0.5) + _GAMMA * _LAM * a1
    assert adv[:, 0] == pytest.approx([a0, a1, a2], rel=1e-6)


# --------------------------------------------------------------------------
# PPO clip loss math
# --------------------------------------------------------------------------


def _loss(logits, values, action, old_logp, adv, ret):
    return ppo_losses(
        torch.tensor(logits),
        torch.tensor(values),
        torch.tensor([action]),
        torch.tensor([old_logp]),
        torch.tensor([adv]),
        torch.tensor([ret]),
    )


def test_ppo_loss_ratio_one():
    # Uniform 2-way logits, ratio=1 => surrogate = -adv; value_loss=0.5*MSE;
    # entropy of uniform-2 = ln 2.
    out = _loss([[0.0, 0.0]], [0.0], action=0, old_logp=-np.log(2), adv=1.0, ret=2.0)
    assert float(out["policy"]) == pytest.approx(-1.0, rel=1e-5)
    assert float(out["value"]) == pytest.approx(0.5 * 4.0, rel=1e-5)
    assert float(out["entropy"]) == pytest.approx(np.log(2), rel=1e-5)
    assert float(out["clip_frac"]) == pytest.approx(0.0)
    assert float(out["approx_kl"]) == pytest.approx(0.0, abs=1e-6)
    # total = policy + 0.5*value - 0.01*entropy
    assert float(out["total"]) == pytest.approx(-1.0 + 0.5 * 2.0 - 0.01 * np.log(2), rel=1e-5)


def test_ppo_loss_clips_positive_advantage():
    # ratio=2 (old_logp = new_logp - ln2), positive advantage => clipped to 1.2.
    new_logp = -np.log(2)
    out = _loss([[0.0, 0.0]], [0.0], action=0, old_logp=new_logp - np.log(2),
                adv=1.0, ret=0.0)
    assert float(out["policy"]) == pytest.approx(-1.2, rel=1e-5)
    assert float(out["clip_frac"]) == pytest.approx(1.0)


def test_ppo_loss_negative_advantage_uses_unclipped():
    # ratio=2, negative advantage => min picks the (more pessimistic) unclipped
    # term -2, so policy loss = +2.
    new_logp = -np.log(2)
    out = _loss([[0.0, 0.0]], [0.0], action=0, old_logp=new_logp - np.log(2),
                adv=-1.0, ret=0.0)
    assert float(out["policy"]) == pytest.approx(2.0, rel=1e-5)


# --------------------------------------------------------------------------
# Rollout buffer shapes + reward placement
# --------------------------------------------------------------------------


def test_rollout_buffer_shapes():
    torch.manual_seed(0)
    model = PolicyNet()
    vec = VecFrameEnv(n=3, base_seed=42)
    gen = torch.Generator().manual_seed(1)
    batch = collect_rollout(model, vec, rollout_len=5, device="cpu", gen=gen)
    B = 5 * 3
    assert batch["obs"].shape == (B, 4, OBS_SIZE, OBS_SIZE)
    assert batch["obs"].dtype == np.uint8
    assert batch["actions"].shape == (B,)
    assert batch["logprobs"].shape == (B,)
    assert batch["values"].shape == (B,)
    assert batch["advantages"].shape == (B,)
    assert batch["returns"].shape == (B,)
    assert batch["rewards"].shape == (5, 3)
    assert batch["dones"].shape == (5, 3)
    assert batch["frames"] == B
    assert set(np.unique(batch["actions"])).issubset(set(range(NUM_ACTIONS)))


def test_reward_lands_on_lock_decision():
    """Scripted single-clear: an O drops into a row that is full except its two
    columns. The clear reward must land on the decision whose tick window
    contains the lock, not the one before it."""
    env = FrameEnv(seed=0)
    env.rows = [0] * 20
    env.rows[19] = FULL_ROW & ~((1 << 3) | (1 << 4))  # bottom row, cols 3,4 open
    env.piece = 1   # O (2x2), cells rows/cols {0,1}x{0,1}
    env.rot = 0
    env.col = 3
    env.row = 18    # O occupies rows 18-19, cols 3-4; floating (gap below-cols empty)
    env.gravity_counter = 20  # descent (=> lock) happens in the SECOND window
    env.tick_count = 0
    env.lines = 0
    env.game_over = False
    env._pending = 0

    r0, d0 = step_decision(env, 0)   # noop; no lock yet
    assert r0 == 0.0 and not d0
    assert env.lines == 0

    r1, d1 = step_decision(env, 0)   # noop; gravity descent locks + clears 1 line
    assert env.lines == 1
    assert r1 == pytest.approx(CLEAR_POINTS[1])  # == 1.0
    assert not d1


def test_terminal_reward_on_game_over():
    # Top out: an O at row -1 whose gravity descent collides with a filled row 1
    # locks with a cell above the board (row < 0) => game over, -10 terminal.
    env = FrameEnv(seed=0)
    env.rows = [0] * 20
    env.rows[1] = FULL_ROW   # blocks the descent so the O locks at row -1
    env.piece = 1            # O (2x2)
    env.rot = 0
    env.col = 0
    env.row = -1             # cells at rows -1,0 (top cell above the board)
    env.gravity_counter = 23  # descent fires on the first tick of this window
    env.tick_count = 0
    env.lines = 0
    env.game_over = False
    env._pending = 0
    r, d = step_decision(env, 0)
    assert d is True
    assert env.game_over
    assert r == pytest.approx(-10.0)  # no clear (row<0 locks before placing) + terminal


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_seeded_rollout_is_deterministic():
    torch.manual_seed(7)
    model = PolicyNet()
    model.eval()

    def run():
        vec = VecFrameEnv(n=4, base_seed=123)
        gen = torch.Generator().manual_seed(999)
        return collect_rollout(model, vec, rollout_len=6, device="cpu", gen=gen)

    b1, b2 = run(), run()
    np.testing.assert_array_equal(b1["actions"], b2["actions"])
    np.testing.assert_array_equal(b1["obs"], b2["obs"])
    np.testing.assert_array_equal(b1["rewards"], b2["rewards"])
    np.testing.assert_array_equal(b1["dones"], b2["dones"])
    np.testing.assert_allclose(b1["advantages"], b2["advantages"], rtol=1e-6)


def test_normalize_advantages_is_sane():
    rng = np.random.default_rng(0)
    adv = rng.normal(3.0, 5.0, size=2048).astype(np.float32)
    norm = normalize_advantages(adv)
    assert abs(float(norm.mean())) < 1e-5
    assert abs(float(norm.std()) - 1.0) < 1e-3


# --------------------------------------------------------------------------
# Time-box exit (patched clock)
# --------------------------------------------------------------------------


def test_time_box_exit_writes_final(tmp_path):
    torch.manual_seed(0)
    model = PolicyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    cfg = {
        "n_envs": 2, "rollout_len": 4, "epochs": 1, "minibatch_size": 8,
        "max_grad_norm": 0.5, "max_hours": 0.01, "max_frames": 10**9,
        "eval_every_frames": 10**9, "eval_games": 1, "eval_max_pieces": 20,
        "seed": 0, "log_every": 1,
    }

    calls = {"n": 0}

    def fake_now():
        # First call is t_start (0.0); everything after is far past the box.
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 1e9

    console = Console(quiet=True)
    with RunWriter("ppo_boxtest", cfg, phase="ppo", root=tmp_path) as run:
        results = train_ppo.run_training(
            run, cfg, model, optimizer, "cpu", console,
            now_fn=fake_now, stop_check=lambda: False,
        )
        assert results["stop_reason"] == "max_hours"
        assert results["updates"] == 1
        assert (run.checkpoints_dir / "nn_final.pt").exists()
        assert results["frames_trained"] == 2 * 4


def test_signal_stop_check_exits(tmp_path):
    torch.manual_seed(0)
    model = PolicyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    cfg = {
        "n_envs": 2, "rollout_len": 4, "epochs": 1, "minibatch_size": 8,
        "max_grad_norm": 0.5, "max_hours": 1e9, "max_frames": 10**9,
        "eval_every_frames": 10**9, "eval_games": 1, "eval_max_pieces": 20,
        "seed": 0, "log_every": 1,
    }
    console = Console(quiet=True)
    with RunWriter("ppo_sigtest", cfg, phase="ppo", root=tmp_path) as run:
        results = train_ppo.run_training(
            run, cfg, model, optimizer, "cpu", console,
            now_fn=lambda: 0.0, stop_check=lambda: True,
        )
        assert results["stop_reason"] == "signal"
        assert results["updates"] == 1
        assert (run.checkpoints_dir / "nn_final.pt").exists()
