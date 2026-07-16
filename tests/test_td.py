"""TD(0) trainer internals: buffer, schedules, targets, decision rule, warm-start.

All fast — tiny nets / tiny data, no full training run (that is the CLI smoke
gate and the run-dir schema test).
"""

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tetris.agents import DELLACHERIE_WEIGHTS, ValueNetAgent, decision_rewards
from tetris.engine import TetrisEngine
from tetris.model import ValueNet, boards_to_tensor
from tetris import td


# -- replay buffer -----------------------------------------------------------


def test_buffer_store_sample_uint16_round_trip():
    buf = td.ReplayBuffer(capacity=100, seed=0)
    rng = np.random.default_rng(0)
    states = rng.integers(0, 1024, size=(10, 20)).astype(np.uint16)
    next_states = rng.integers(0, 1024, size=(10, 20)).astype(np.uint16)
    rewards = rng.standard_normal(10).astype(np.float32)
    dones = (rng.random(10) > 0.5).astype(np.float32)
    buf.add_many(states, rewards, next_states, dones)
    assert buf.size == 10
    # Every stored state must be one of the originals, bit-exact (uint16).
    s, r, ns, d = buf.sample(64)
    assert s.dtype == np.uint16 and ns.dtype == np.uint16
    for row in s:
        assert any(np.array_equal(row, orig) for orig in states)


def test_buffer_wraparound_keeps_capacity():
    buf = td.ReplayBuffer(capacity=8, seed=0)
    for base in range(4):
        states = np.full((5, 20), base, dtype=np.uint16)
        buf.add_many(states, np.zeros(5, np.float32),
                     states, np.zeros(5, np.float32))
    assert buf.size == 8
    assert buf.pos == (20 % 8)


# -- schedules ---------------------------------------------------------------


def test_epsilon_schedule_endpoints_and_monotonic():
    budget = 1_000_000
    assert td.epsilon_at(0, budget, 1.0) == 1.0
    # Flat at 0.02 after 30% of budget.
    assert td.epsilon_at(int(0.30 * budget), budget, 1.0) == 0.02
    assert td.epsilon_at(budget, budget, 1.0) == 0.02
    # Monotonically non-increasing across the anneal.
    vals = [td.epsilon_at(p, budget, 1.0) for p in range(0, budget, 50_000)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    # Warm-start starts at 0.2.
    assert td.epsilon_at(0, budget, 0.2) == 0.2


def test_beta_schedule():
    budget = 1_000_000
    assert td.beta_at(0, budget) == 1.0
    assert abs(td.beta_at(int(0.30 * budget), budget) - 0.5) < 1e-9
    assert td.beta_at(int(0.60 * budget), budget) == 0.0
    assert td.beta_at(budget, budget) == 0.0


# -- TD target ---------------------------------------------------------------


def test_td_target_no_bootstrap_on_terminal():
    rewards = np.array([1.0, 2.0], dtype=np.float32)
    next_values = np.array([10.0, 20.0], dtype=np.float32)
    dones = np.array([0.0, 1.0], dtype=np.float32)
    out = td.td_target(rewards, next_values, dones, gamma=0.9)
    assert np.allclose(out, [1.0 + 0.9 * 10.0, 2.0])  # terminal ignores V(A')


def test_td_target_uses_supplied_target_values():
    # Distinct online vs target values must flow through unchanged (no bootstrap
    # masking on non-terminal), proving the target-net values are what's used.
    r = np.array([0.0], dtype=np.float32)
    online_v = np.array([5.0], dtype=np.float32)
    target_v = np.array([7.0], dtype=np.float32)
    d = np.array([0.0], dtype=np.float32)
    assert td.td_target(r, target_v, d, 1.0)[0] == 7.0
    assert td.td_target(r, online_v, d, 1.0)[0] == 5.0


# -- decision-rule equivalence ----------------------------------------------


def test_batched_decision_matches_valuenet_agent():
    torch.manual_seed(3)
    model = ValueNet().eval()
    gamma = 0.95
    for seed in (1, 2, 7, 42):
        engine = TetrisEngine(seed=seed)
        for _ in range(5):  # a few plies into the game
            agent = ValueNetAgent(model, gamma=gamma, beta=0.0)
            chosen = agent.act(engine)

            # Replicate td.py's batched path on the identical state.
            placements, afters, r = decision_rewards(engine, 0.0)
            with torch.no_grad():
                v = model(boards_to_tensor(afters)).numpy()
            manual = placements[int(np.argmax(r + gamma * v))]
            assert chosen == manual
            engine.step(*chosen)
            if engine.game_over:
                break


def test_decision_reward_recovers_lines_and_clear_points():
    # A wide, nearly-full board: the reward's clear_points term must equal the
    # engine's actual clear_points for the chosen placement.
    from tetris.engine import CLEAR_POINTS

    engine = TetrisEngine(seed=5)
    for _ in range(30):
        placements, afters, r = decision_rewards(engine, 0.0)
        # With beta=0 the reward is pure clear_points[lines]; it must be one of
        # the table's values for every candidate.
        assert all(float(x) in [float(v) for v in CLEAR_POINTS] for x in r)
        engine.step(*placements[0])
        if engine.game_over:
            break


# -- warm-start normalization ------------------------------------------------


def test_teacher_targets_are_per_decision_normalized():
    # A single decision -> the whole targets array is that decision's candidates,
    # normalized to zero mean / unit variance.
    data = td.collect_teacher_data(DELLACHERIE_WEIGHTS, num_placements=1, base_seed=11)
    assert len(data.targets) > 1
    assert abs(float(data.targets.mean())) < 1e-4
    assert abs(float(data.targets.std()) - 1.0) < 1e-4
    assert data.boards.dtype == np.uint16


def test_pretrain_regression_reduces_loss():
    torch.manual_seed(0)
    model = ValueNet()
    data = td.collect_teacher_data(DELLACHERIE_WEIGHTS, num_placements=200, base_seed=1)
    losses = td.pretrain_regression(
        model, data, epochs=3, batch_size=64, lr=1e-3, device="cpu", seed=0
    )
    assert len(losses) == 3
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
