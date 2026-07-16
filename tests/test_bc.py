"""BC dataset tests (PLAN2.md §5, Phase C — dataset half).

Covers: obs bit-pack round-trip, class histogram + capped inverse-frequency
weights, end-to-end tiny dataset generation with byte-exact frame round-trip vs
a fresh expert replay, 4-stack reconstruction (including the episode-boundary
first-frame-repeat rule), and the linear-probe gate (slow): a ridge regressor on
10k observations recovers per-column stack heights with <= 0.5 MAE, proving the
render carries the board state.

The CEM teacher is used throughout (torch-free, fast), matching the keypress-
expert unit tests; the render carries state regardless of which teacher plays.
"""

import numpy as np
import pytest

from tetris.bc import (
    BCDataset,
    class_histogram,
    column_heights,
    generate_dataset,
    inverse_freq_weights,
    pack_obs,
    unpack_obs,
)
from tetris.engine import PIECES
from tetris.frame_env import FrameEnv
from tetris.keypress_expert import ExpertPlayer, make_teacher
from tetris.render_obs import render_env


# -- packing ---------------------------------------------------------------


def test_pack_unpack_roundtrip():
    env = FrameEnv(seed=3)
    player = ExpertPlayer(make_teacher("cem"))
    player.reset(env)
    for _ in range(600):  # build a stack so the frame is non-trivial
        if env.game_over:
            break
        if env.is_decision_tick:
            env.apply_action(player.act(env))
        env.tick()
    obs = render_env(env)
    packed = pack_obs(obs)
    assert packed.shape == (1152,)
    assert packed.dtype == np.uint8
    np.testing.assert_array_equal(unpack_obs(packed), obs)


# -- class weighting -------------------------------------------------------


def test_class_histogram_and_weights():
    actions = np.array([0] * 90 + [1] * 9 + [2] * 1, dtype=np.uint8)  # no 3/4
    hist = class_histogram(actions)
    assert hist.tolist() == [90, 9, 1, 0, 0]

    w = inverse_freq_weights(hist, cap=20.0)
    assert w[0] == pytest.approx(1.0)          # majority class -> 1.0
    assert w[1] == pytest.approx(10.0)         # 90/9
    assert w[2] == pytest.approx(20.0)         # 90/1 = 90, capped at 20
    assert w[3] == pytest.approx(20.0)         # absent -> cap
    assert w[4] == pytest.approx(20.0)
    assert np.all(w <= 20.0) and np.all(w >= 1.0)


# -- dataset generation + round-trip + stack reconstruction ----------------


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("bc_data")
    meta = generate_dataset(
        out_dir=out,
        total_pieces=40,
        max_game_pieces=15,   # forces several short episodes
        base_seed=555000,
        teacher_kind="cem",
        progress=False,
    )
    return out, meta


def test_dataset_shapes_and_consistency(tiny_dataset):
    out, meta = tiny_dataset
    ds = BCDataset(out)
    n = meta["n_frames"]
    assert len(ds) == n
    assert ds.frames.shape == (n, 1152)
    assert ds.actions.shape == (n,)
    assert ds.episode_id.shape == (n,)
    assert n == sum(e[1] for e in meta["episodes"])
    assert meta["n_episodes"] >= 2  # multiple episodes for the boundary test
    # Class weights present for every action.
    assert set(meta["class_weights"]) == set(meta["actions"])


def test_frame_roundtrip_vs_replay(tiny_dataset):
    # Episode 0 stored frames/actions must byte-match a fresh deterministic
    # replay of its seed with the same expert (proves the writer stored exactly
    # what render_env produced, in order).
    out, meta = tiny_dataset
    ds = BCDataset(out)
    start, length, seed, *_ = meta["episodes"][0]
    env = FrameEnv(seed=seed)
    player = ExpertPlayer(make_teacher("cem"))
    player.reset(env)
    i = start
    produced = 0
    while not env.game_over and env.pieces < meta["max_game_pieces"]:
        if env.is_decision_tick:
            obs = render_env(env)
            np.testing.assert_array_equal(ds.frame(i), obs)
            a = player.act(env)
            assert ds.actions[i] == a
            env.apply_action(a)
            i += 1
            produced += 1
        env.tick()
    assert produced == length


def test_stack_reconstruction_and_boundary(tiny_dataset):
    out, meta = tiny_dataset
    ds = BCDataset(out)
    for start, length, *_ in meta["episodes"]:
        # First frame of an episode: stack repeats it four times.
        assert ds.stack_indices(start) == [start, start, start, start]
        if length > 3:
            assert ds.stack_indices(start + 1) == [start, start, start, start + 1]
            assert ds.stack_indices(start + 2) == [start, start, start + 1, start + 2]
            i = start + 3
            assert ds.stack_indices(i) == [start, start + 1, start + 2, i]
    # Stack tensor: (4,96,96) float in {0,1}, last slice == normalized frame(i).
    i = min(meta["n_frames"] - 1, meta["episodes"][0][0] + meta["episodes"][0][1] - 1)
    st = ds.stack(i)
    assert st.shape == (4, 96, 96)
    assert st.dtype == np.float32
    assert set(np.unique(st).tolist()) <= {0.0, 1.0}
    np.testing.assert_array_equal((st[3] * 255).astype(np.uint8), ds.frame(i))


# -- linear-probe gate (slow) ----------------------------------------------


def _piece_above_board(env) -> bool:
    """True iff every active-piece cell is above the visible board (row+ro < 0),
    so the observation's board region shows only the locked stack."""
    return all(env.row + ro < 0 for ro, _ in PIECES[env.piece][env.rot].cells)


@pytest.mark.slow
def test_linear_probe_recovers_heights():
    # Gate: a ridge regressor trained on 10k observations recovers per-column
    # stack heights with <= 0.5 MAE (PLAN2.md §5). We probe REAL observations at
    # decision ticks where the active piece is still entirely above the board, so
    # the pixels the probe reads encode the stack (the active piece and stack are
    # deliberately indistinguishable in the render, which otherwise confounds a
    # pure stack-height readout). Full 96x96 pixels; held-out MAE. Runs in ~5s.
    teacher = make_teacher("cem")
    player = ExpertPlayer(teacher)
    N = 10000
    X = np.empty((N, 96 * 96), dtype=np.float32)
    Y = np.empty((N, 10), dtype=np.float32)
    k = 0
    ep = 0
    while k < N:
        env = FrameEnv(seed=700000 + ep)
        player.reset(env)
        ep += 1
        while not env.game_over and k < N:
            if env.is_decision_tick:
                if _piece_above_board(env):
                    X[k] = render_env(env).reshape(-1) / 255.0
                    Y[k] = column_heights(env.rows)
                    k += 1
                env.apply_action(player.act(env))
            env.tick()

    ntr = 8000
    d = X.shape[1]
    A = X[:ntr].T @ X[:ntr] + 1.0 * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(A, X[:ntr].T @ Y[:ntr])
    mae = float(np.mean(np.abs(X[ntr:] @ W - Y[ntr:])))
    assert mae <= 0.5, f"linear-probe height MAE {mae:.3f} exceeds 0.5"
