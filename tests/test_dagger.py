"""DAgger relabel + closed-loop policy eval tests (PLAN2.md §6, Phase D).

The critical Phase B carry-over: the expert streams spawn-time scripts and cannot
relabel arbitrary mid-flight states, so DAgger must replan from the CURRENT pose.
These tests pin that:

* :func:`current_pose_script` equals :func:`naive_script` at spawn but produces
  the correct actions from a moved pose, where the spawn-time script mislabels.
* :func:`relabel_action` always returns the first action of a current-pose script
  that forward-sim confirms lands at the chosen placement (``tuck=False``) — i.e.
  the label matches forward-sim reality — and diverges from the naive spawn-script
  label on real mid-flight states.

Plus the shared closed-loop eval, batch stack reconstruction, and the
MultiBCDataset aggregation used to fold DAgger data into training.

The CEM teacher is used throughout (torch-free, fast).
"""

import numpy as np
import pytest

from tetris.engine import PIECES
from tetris.features import WIDTH
from tetris.frame_env import LEFT, NOOP, FrameEnv
from tetris.keypress_expert import (
    DaggerRelabeler,
    ExpertPlayer,
    clone_env,
    current_pose_script,
    fully_visible,
    make_teacher,
    naive_script,
    relabel_action,
    simulate_script,
)

I, O, T, S, Z, J, L = range(7)


def _cem():
    return make_teacher("cem")


# -- current-pose script vs naive spawn-script -----------------------------


def test_current_pose_equals_naive_at_spawn():
    # From the spawn pose (rot 0, spawn column) the current-pose script is the
    # naive spawn script for every piece/target — the general case reduces to it.
    for piece in range(7):
        spawn_col = (WIDTH - PIECES[piece][0].width) // 2
        n = len(PIECES[piece])
        for tr in range(n):
            for tc in range(WIDTH):
                assert (current_pose_script(piece, 0, spawn_col, tr, tc)
                        == naive_script(piece, tr, tc))


def test_midflight_script_lands_where_naive_cannot():
    # Hand-built mid-flight state: empty board, I-piece slid to col 6, descended.
    # Target col 3 needs 3 LEFTs from HERE; the spawn-time naive script (spawn
    # col == 3) is EMPTY and would leave it at col 6 — the wrong placement.
    env = FrameEnv(seed=0)
    env.rows = [0] * 20
    env.piece = I
    env.rot, env.col, env.row = 0, 6, 10
    env.tick_count, env.gravity_counter, env.game_over = 0, 0, False

    cps = current_pose_script(env.piece, env.rot, env.col, 0, 3)
    assert cps == [LEFT, LEFT, LEFT]
    lock = simulate_script(clone_env(env), cps)
    assert (lock["r"], lock["c"], lock["tuck"]) == (0, 3, False)

    # Naive spawn-assumption script mislands from this mid-flight pose.
    nav = naive_script(env.piece, 0, 3)
    assert nav == []  # spawn col already 3
    nav_lock = simulate_script(clone_env(env), nav)
    assert nav_lock["c"] == 6  # ends at the CURRENT column, not the target


# -- relabel truthfulness (matches forward-sim reality) --------------------


def _chosen_placement(env, teacher):
    """Re-run relabel's search to recover the placement it selected (highest-
    scored placement reachable from the current pose), or None."""
    scored = teacher.scores(_pe(env))
    if scored is None:
        return None
    placements, scores = scored
    for idx in np.argsort(-scores, kind="stable"):
        rot, col = placements[int(idx)]
        script = current_pose_script(env.piece, env.rot, env.col, rot, col)
        lock = simulate_script(clone_env(env), script)
        if lock is not None and lock["r"] == rot and lock["c"] == col and not lock["tuck"]:
            return rot, col, script
    return None


def _pe(env):
    from tetris.keypress_expert import _placement_engine
    return _placement_engine(env)


def test_relabel_matches_forward_sim_and_diverges_from_naive():
    teacher = _cem()
    verified = 0
    diverged = 0
    invisible = 0
    for seed in range(12):
        env = FrameEnv(seed=seed)
        player = ExpertPlayer(teacher)
        player.reset(env)
        while not env.game_over and env.pieces < 60:
            if env.is_decision_tick:
                a = relabel_action(env, teacher)
                if not fully_visible(env):
                    # §4 amendment: no label before the camera sees the piece.
                    assert a == NOOP
                    invisible += 1
                else:
                    chosen = _chosen_placement(env, teacher)
                    if chosen is not None:
                        rot, col, script = chosen
                        # Label = first action of the forward-sim-verified script.
                        assert a == (script[0] if script else NOOP)
                        lock = simulate_script(clone_env(env), script)
                        assert (lock["r"], lock["c"], lock["tuck"]) == (rot, col, False)
                        verified += 1
                        # Divergence: naive spawn-script first action for the
                        # same target vs the current-pose label.
                        spawn_col = (WIDTH - PIECES[env.piece][0].width) // 2
                        if env.rot != 0 or env.col != spawn_col:
                            nav = naive_script(env.piece, rot, col)
                            nav_first = nav[0] if nav else NOOP
                            if nav_first != a:
                                diverged += 1
                env.apply_action(player.act(env))
            env.tick()
    assert verified > 500, f"too few relabels verified ({verified})"
    assert diverged > 0, "expected mid-flight relabel/naive divergences"
    assert invisible > 100, "expected above-board decision ticks in real games"


def test_relabel_noop_while_not_fully_visible():
    teacher = _cem()
    # Fresh spawn: bounding-box bottom at board row -1 => not fully visible.
    env = FrameEnv(seed=7)
    assert env.is_decision_tick and not fully_visible(env)
    assert relabel_action(env, teacher) == NOOP
    assert DaggerRelabeler(teacher).relabel(env) == NOOP
    # Hand-built PARTIALLY visible state: T (2 rows tall) straddling the top
    # edge — bottom cells on row 0 are rendered, top cells at row -1 are not.
    env2 = FrameEnv(seed=0)
    env2.rows = [0] * 20
    env2.piece, env2.rot, env2.col, env2.row = T, 0, 3, -1
    env2.tick_count, env2.gravity_counter, env2.game_over = 0, 0, False
    assert not fully_visible(env2)
    assert relabel_action(env2, teacher) == NOOP
    # One row lower it is fully visible and the relabel engages.
    env2.row = 0
    assert fully_visible(env2)


def test_relabel_matches_player_at_first_visible_tick():
    # Tick a fresh game to the FIRST fully-visible decision tick (piece still at
    # rot 0 / spawn col — only descended). The relabel's current-pose replan and
    # the player's spawn-time plan see the same board, pose, and reachability
    # (spawn-time sims wait for visibility too), so their actions must agree.
    teacher = _cem()
    env = FrameEnv(seed=7)
    player = ExpertPlayer(teacher)
    player.reset(env)
    while True:
        if env.is_decision_tick and fully_visible(env):
            a_relabel = relabel_action(env, teacher)
            a_player = player.act(env)
            assert a_relabel == a_player
            break
        if env.is_decision_tick:
            env.apply_action(player.act(env))  # NOOPs while invisible
        env.tick()


def test_relabeler_cache_matches_uncached():
    # The per-piece score cache in DaggerRelabeler must not change any label vs
    # the stateless relabel_action.
    teacher = _cem()
    env = FrameEnv(seed=3)
    player = ExpertPlayer(teacher)
    player.reset(env)
    relab = DaggerRelabeler(teacher)
    while not env.game_over and env.pieces < 40:
        if env.is_decision_tick:
            assert relab.relabel(env) == relabel_action(env, teacher)
            env.apply_action(player.act(env))
        env.tick()


# -- batch stack reconstruction + multi-dataset ----------------------------


@pytest.fixture(scope="module")
def tiny_ds(tmp_path_factory):
    from tetris.bc import generate_dataset
    out = tmp_path_factory.mktemp("bc_batch")
    generate_dataset(out_dir=out, total_pieces=30, max_game_pieces=8,
                     base_seed=321000, teacher_kind="cem", progress=False)
    return out


def test_batch_stacks_matches_single(tiny_ds):
    from tetris.bc import BCDataset
    ds = BCDataset(tiny_ds)
    idx = np.array([0, 1, 2, 3, 5, len(ds) - 1, len(ds) // 2])
    batch = ds.batch_stacks(idx)
    assert batch.shape == (len(idx), 4, 96, 96)
    for bi, i in enumerate(idx):
        np.testing.assert_array_equal(batch[bi], ds.stack(int(i)))


def test_multi_dataset_indexing(tiny_ds):
    from tetris.bc import BCDataset, MultiBCDataset
    a = BCDataset(tiny_ds)
    b = BCDataset(tiny_ds)
    multi = MultiBCDataset([a, b])
    assert len(multi) == 2 * len(a)
    assert multi.actions.shape == (2 * len(a),)
    # Global indices spanning both shards reconstruct within-shard stacks.
    idx = np.array([0, 3, len(a) - 1, len(a), len(a) + 3, 2 * len(a) - 1])
    batch = multi.batch_stacks(idx)
    for bi, gi in enumerate(idx):
        local = gi if gi < len(a) else gi - len(a)
        src = a if gi < len(a) else b
        np.testing.assert_array_equal(batch[bi], src.stack(int(local)))


# -- class-balanced batch sampler (§6 amendment) ----------------------------


def test_balanced_batches_uniformish(tiny_ds):
    from tetris.bc import BCDataset, balanced_batches

    ds = BCDataset(tiny_ds)
    rng = np.random.default_rng(0)
    present = [a for a in range(5) if (ds.actions == a).any()]
    k = len(present)
    batches = list(balanced_batches(ds, 65, 4, rng))
    assert len(batches) == 4  # yields exactly total_steps batches
    for stacks, labels in batches:
        assert stacks.shape == (65, 4, 96, 96)
        assert labels.shape == (65,) and labels.dtype == np.int64
        counts = np.bincount(labels, minlength=5)
        for a in range(5):
            if a in present:
                # each present class gets its ~uniform share (base or base+1)
                assert abs(counts[a] - 65 / k) <= 1
            else:
                assert counts[a] == 0


def test_balanced_batches_labels_match_frames(tiny_ds):
    # Every sampled (stack, label) pair must be a real dataset pair: the last
    # slice of the stack equals the frame whose action is the label, for some
    # dataset index with that exact frame+action. Verify via lookup by content.
    from tetris.bc import BCDataset, balanced_batches, pack_obs

    ds = BCDataset(tiny_ds)
    key_to_actions = {}
    for i in range(len(ds)):
        key_to_actions.setdefault(ds.frames[i].tobytes(), set()).add(int(ds.actions[i]))
    rng = np.random.default_rng(1)
    stacks, labels = next(balanced_batches(ds, 32, 1, rng))
    for j in range(32):
        frame = (stacks[j, 3] * 255).astype(np.uint8)
        key = pack_obs(frame).tobytes()
        assert int(labels[j]) in key_to_actions[key]


# -- closed-loop policy eval (shared with PPO) -----------------------------


def test_evaluate_policy_untrained_runs():
    import torch

    from tetris.bc import evaluate_policy
    from tetris.policy_model import PolicyNet

    torch.manual_seed(0)
    model = PolicyNet()
    ev = evaluate_policy(model, seeds=[950000, 950001, 950002], max_pieces=30,
                         device="cpu")
    assert len(ev.lines) == 3 and len(ev.pieces) == 3
    assert all(p <= 30 for p in ev.pieces)
    assert ev.median_lines >= 0.0
    assert 0 <= ev.best_index < 3
    # A move list was recorded for the best game (physical (r, c) locks).
    assert isinstance(ev.moves[ev.best_index], list)


def test_dagger_rollout_and_write(tmp_path):
    import torch

    from tetris.bc import (BCDataset, MultiBCDataset, dagger_rollout,
                           write_dagger_dataset)
    from tetris.policy_model import PolicyNet

    torch.manual_seed(0)
    model = PolicyNet()
    roll = dagger_rollout(model, _cem(), target_frames=200, base_seed=888000,
                          device="cpu", max_pieces=40)
    assert roll["n_frames"] >= 200
    assert len(roll["packed"]) == roll["n_frames"]
    assert roll["actions"].shape == (roll["n_frames"],)
    d = write_dagger_dataset(tmp_path / "dagger0", roll)
    ds = BCDataset(d)
    assert len(ds) == roll["n_frames"]
    # Labels are valid actions and stacks reconstruct.
    assert ds.actions.min() >= 0 and ds.actions.max() <= 4
    b = ds.batch_stacks(np.arange(min(16, len(ds))))
    assert b.shape[1:] == (4, 96, 96)
