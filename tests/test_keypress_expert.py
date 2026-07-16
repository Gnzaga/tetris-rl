"""Keypress-expert tests (PLAN2.md §4).

Covers: naive-script validity (forward-sim lands the predicted (r, c) with no
tuck on many seeds/pieces), determinism (same seed => identical game + action
stream), unreachability detection on a hand-built deep-well case, and an
ExpertPlayer end-to-end short game. The CEM teacher is used throughout so the
unit tests stay torch-free and fast; the ValueNet teacher is exercised by the
Phase B eval gate.
"""

import pytest

from tetris.features import FULL_ROW
from tetris.frame_env import NOOP, FrameEnv
from tetris.keypress_expert import (
    ExpertPlayer,
    _placement_engine,
    clone_env,
    is_reachable,
    make_teacher,
    naive_script,
    plan,
    simulate_script,
)

# Piece indices (shared/pieces.json order): I O T S Z J L
I, O, T, S, Z, J, L = range(7)


def _cem():
    return make_teacher("cem")


def _drive_verify(seed, teacher, max_pieces):
    """Play a game with the expert; at every new-piece plan, forward-sim the
    chosen script on a clone and assert it locks at the planned (r, c) with no
    tuck (whenever the plan actually found a reachable placement). Returns the
    number of plans verified."""
    env = FrameEnv(seed=seed)
    player = ExpertPlayer(teacher)
    player.reset(env)
    verified = 0
    while not env.game_over and env.pieces < max_pieces:
        if env.is_decision_tick:
            new_piece = env.pieces != player._plan_pieces
            action = player.act(env)  # plans on a new piece; does not mutate env
            if new_piece:
                pr = player.last_plan
                if pr.target is not None and pr.reachable:
                    lock = simulate_script(clone_env(env), pr.script)
                    assert lock is not None
                    assert (lock["r"], lock["c"]) == pr.target
                    assert lock["tuck"] is False
                    verified += 1
            env.apply_action(action)
        env.tick()
    return verified


# -- script validity --------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 7, 42, 900000, 123456, 2024, 999])
def test_script_lands_predicted_pose(seed):
    verified = _drive_verify(seed, _cem(), max_pieces=120)
    assert verified >= 50  # a healthy game plans many reachable pieces


def test_naive_script_shapes():
    # I: rot0 width4 col3. Vertical (rot1) needs 1 cw then slide.
    s = naive_script(I, 1, 5)
    assert s[0] == 3  # ROT_CW
    # O has a single rotation state: never emits a rotation.
    from tetris.keypress_expert import ROT_CW, ROT_CCW

    assert all(a not in (ROT_CW, ROT_CCW) for a in naive_script(O, 0, 7))


# -- determinism ------------------------------------------------------------


def test_determinism_same_seed_same_game():
    def run(seed):
        env = FrameEnv(seed=seed)
        player = ExpertPlayer(_cem())
        player.reset(env)
        actions = []
        while not env.game_over and env.pieces < 60:
            if env.is_decision_tick:
                a = player.act(env)
                actions.append(a)
                env.apply_action(a)
            env.tick()
        return env.lines, env.pieces, actions

    a = run(900000)
    b = run(900000)
    assert a == b
    # A different seed should generally diverge (different piece stream).
    assert run(900001)[2] != a[2]


# -- unreachability detection ----------------------------------------------


def _deep_well_env():
    """Cols 0-8 filled for rows 1..19, col 9 an empty full-depth well; I piece
    at spawn. Vertical I in the well (rot 1, col 9) is a legal v1 straight drop
    but unreachable: rotating to vertical near spawn collides with the wall, so
    the naive rotate-then-slide script can never seat it in the well."""
    env = FrameEnv(seed=0)
    env.rows = [0] * 20
    wall = FULL_ROW & ~(1 << 9)  # columns 0-8 set, column 9 clear
    for r in range(1, 20):
        env.rows[r] = wall
    env.piece = I
    env.rot, env.col, env.row = 0, 3, -1
    env.tick_count = 0
    env.gravity_counter = 0
    env.game_over = False
    env._pending = NOOP
    return env


def test_unreachable_deep_well():
    env = _deep_well_env()
    eng = _placement_engine(env)
    placements, _, _ = eng.candidate_features()
    assert (1, 9) in placements  # legal v1 straight drop into the well
    # The deep-well vertical placement is unreachable...
    assert is_reachable(env, 1, 9) is False
    # ...while the shallow horizontal placements on top are reachable.
    assert is_reachable(env, 0, 0) is True
    assert is_reachable(env, 0, 6) is True


def test_plan_avoids_unreachable_top():
    # With the well board, force the well placement to look best by using a
    # teacher, then confirm plan never returns an unreachable target.
    env = _deep_well_env()
    pr = plan(env, _cem(), check_all_reachable=True)
    assert pr.target is not None
    assert pr.reachable is True
    assert is_reachable(env, *pr.target) is True
    n_reach, n_total = pr.reachable_all
    assert 0 < n_reach < n_total  # the well placement is counted unreachable


# -- ExpertPlayer end-to-end ------------------------------------------------


def test_expert_player_short_game():
    env = FrameEnv(seed=900000)
    player = ExpertPlayer(_cem())
    player.reset(env)
    locks = 0
    while not env.game_over and env.pieces < 40:
        if env.is_decision_tick:
            action = player.act(env)
            env.apply_action(action)
        if env.tick() is not None:
            locks += 1
    assert env.pieces == locks
    assert env.pieces >= 30  # clears lines, survives well past spawn
    assert env.lines > 0


def test_act_streams_noop_after_script():
    # After a plan's script is exhausted the player must emit NOOP until lock.
    env = FrameEnv(seed=900000)
    player = ExpertPlayer(_cem())
    player.reset(env)
    player.act(env)  # plans piece 0
    script_len = len(player._script)
    # Advance the internal index past the script without changing pieces.
    player._i = script_len
    assert player.act(env) == NOOP
