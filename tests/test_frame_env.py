"""Frame-layer unit tests (PLAN2.md §3, amended §1: physical-pose locking).

Covers gravity / lock / spawn semantics on hand-built scenarios, the rotation
clamp and its failure cases, line clears + scoring, above-board locks ending
the game with the board unchanged, frame-layer determinism, the straight-drop
invariant (drop row <= lock row), a constructed tuck locking at its physical
pose with the exact post-lock board, and the v1-consistency invariant: every
non-tuck lock transition is bit-identical to v1 engine.step(r, c), cross-checked
by a parallel bare v1 engine over the non-tuck prefix of random games.
"""

import pytest

from tetris.engine import PIECES, TetrisEngine
from tetris.features import FULL_ROW, HEIGHT, WIDTH
from tetris.frame_env import (
    DECISION_PERIOD,
    GRAVITY_PERIOD,
    LEFT,
    RIGHT,
    ROT_CW,
    FrameEnv,
)
from tetris.rng import Mulberry32

# Piece indices (shared/pieces.json order): I O T S Z J L
I, O, T, S, Z, J, L = range(7)


def test_spawn_pose():
    # Spawn: rot 0, col floor((10-width)/2), bbox bottom at board row -1.
    env = FrameEnv(seed=0)
    env.piece = I  # width 4, height 1
    env._spawn()
    assert env.pose() == (I, 0, 3, -1)  # -height = -1

    env.piece = O  # width 2, height 2
    env._spawn()
    assert env.pose() == (O, 0, 4, -2)

    env.piece = T  # width 3, height 2
    env._spawn()
    assert env.pose() == (T, 0, 3, -2)


def test_piece_supply_matches_v1_engine():
    # Same seed => same current piece and preview queue as the v1 engine.
    for seed in (0, 7, 123456):
        env = FrameEnv(seed=seed)
        eng = TetrisEngine(seed=seed)
        assert env.piece == eng.current
        assert env.preview_pieces() == eng.preview_pieces()


def test_decision_and_gravity_cadence():
    env = FrameEnv(seed=0)
    # Decision ticks at multiples of DECISION_PERIOD.
    assert env.is_decision_tick
    for t in range(10):
        assert env.is_decision_tick == (t % DECISION_PERIOD == 0)
        env.tick()
    # Gravity: after exactly GRAVITY_PERIOD ticks the piece has descended 1 row.
    env2 = FrameEnv(seed=0)
    start_row = env2.row
    for _ in range(GRAVITY_PERIOD - 1):
        env2.tick()
    assert env2.row == start_row  # not yet
    env2.tick()  # the GRAVITY_PERIOD-th tick
    assert env2.row == start_row + 1
    assert env2.gravity_counter == 0


def test_apply_action_rejected_off_decision_tick():
    env = FrameEnv(seed=0)
    env.tick()  # tick_count -> 1, not a decision tick
    assert not env.is_decision_tick
    with pytest.raises(ValueError):
        env.apply_action(LEFT)


def test_slide_moves_and_wall_noop():
    env = FrameEnv(seed=0)
    env.piece = I  # spawn col 3, width 4 -> col range [0, 6]
    env._spawn()
    # Slide left twice on consecutive decision ticks.
    env.apply_action(LEFT)
    env.tick()  # decision tick 0
    assert env.col == 2
    # advance to next decision tick
    env.tick()
    env.tick()
    assert env.is_decision_tick
    env.apply_action(LEFT)
    env.tick()
    assert env.col == 1

    # Push to the left wall then attempt one more: silent no-op.
    env.piece = I
    env._spawn()
    env.col = 0
    env.tick_count = 0  # force decision tick
    env.apply_action(LEFT)
    env.tick()
    assert env.col == 0  # wall blocked, silent no-op

    # Right wall: I at col 6 (max), sliding right is a no-op.
    env.piece = I
    env._spawn()
    env.col = 6
    env.tick_count = 0
    env.apply_action(RIGHT)
    env.tick()
    assert env.col == 6


def test_slide_blocked_by_stack():
    env = FrameEnv(seed=0)
    env.piece = O  # width 2, height 2, spawn col 4
    env._spawn()
    env.row = 5  # bring it onto the board
    # Fill the cell to the right so a right-slide collides with the stack.
    # O at (rot0, col4, row5) covers cols 4,5 rows 5,6. Right slide -> cols 5,6.
    env.rows[5] = 1 << 6  # col6 filled at row5
    env.tick_count = 0
    env.apply_action(RIGHT)
    env.tick()
    assert env.col == 4  # blocked by stack cell at (5, 6)


def test_rotation_clamp_at_right_wall():
    # I piece rotated to vertical (width 1) then back to horizontal (width 4)
    # near the right wall must clamp the column into [0, 10-4] = [0, 6].
    env = FrameEnv(seed=0)
    env.piece = I
    env._spawn()
    env.rot = 1  # vertical, width 1, columns [0,9]
    env.col = 9  # far right, valid for width 1
    env.row = 0
    env.tick_count = 0
    env.apply_action(ROT_CW)  # back to horizontal width 4
    env.tick()
    assert env.rot == 0
    assert env.col == 6  # clamped from 9 to 10-4


def test_rotation_fails_on_collision():
    # A rotation whose clamped cells collide with the stack is a silent no-op.
    env = FrameEnv(seed=0)
    env.piece = I
    env._spawn()
    env.rot = 1  # vertical, occupies one column, rows row..row+3
    env.col = 5
    env.row = 16  # cells at rows 16..19 in col 5
    # Fill the row that the horizontal rotation would occupy so it collides.
    env.rows[16] = (1 << WIDTH) - 1  # entire row 16 filled
    env.tick_count = 0
    env.apply_action(ROT_CW)
    env.tick()
    assert env.rot == 1  # rotation failed, unchanged
    assert env.col == 5


def test_lock_on_floor_and_v1_consistency():
    # Drop an I piece straight to the floor with no actions; it must lock at the
    # straight-drop position and the transition must be bit-identical to a bare
    # v1 engine.step of the same (r, c).
    env = FrameEnv(seed=0)
    env.piece = I
    env._spawn()  # col 3, row -1
    locks = []
    # Enough ticks for the I to fall 20 rows (20 * 24 = 480) and lock.
    for _ in range(GRAVITY_PERIOD * 22):
        lk = env.tick()
        if lk is not None:
            locks.append(lk)
            break
    assert len(locks) == 1
    lock = locks[0]
    assert lock["r"] == 0 and lock["c"] == 3
    assert lock["tuck"] is False
    # Bottom row filled at cols 3..6.
    assert env.rows[HEIGHT - 1] == 0b0000_1111_000  # cols 3,4,5,6
    assert env.pieces == 1

    # v1-consistency: the same placement through a bare v1 engine.
    bare = TetrisEngine(seed=0)
    bare.current = I
    bare.step(lock["r"], lock["c"])
    assert bare.rows == env.rows


def test_line_clear_and_score():
    # Bottom row full except cols 3..6; a flat I locks there and clears it.
    env = FrameEnv(seed=0)
    env.piece = I
    env._spawn()  # rot 0, col 3
    env.rows[HEIGHT - 1] = FULL_ROW ^ (0b1111 << 3)
    lock = None
    for _ in range(GRAVITY_PERIOD * 22):
        lk = env.tick()
        if lk is not None:
            lock = lk
            break
    assert lock is not None
    assert lock["lines_after"] == 1
    assert env.lines == 1
    assert env.score == 100  # CLEAR_POINTS[1] * 100
    assert env.rows == [0] * HEIGHT  # cleared row shifted away, board empty


def test_constructed_tuck_locks_at_physical_pose():
    # A vertical I slides under an overhang and locks BELOW the straight-drop
    # row, at its true physical pose (amended §1: what the camera sees is what
    # locks). Exact post-lock board asserted.
    env = FrameEnv(seed=0)
    env.piece = I
    env._spawn()
    env.rows = [0] * HEIGHT
    env.rows[10] = 1 << 1  # overhang: col1 filled at row 10, empty below
    env.rot = 1  # vertical I, width 1
    env.col = 0  # open column
    env.row = 15  # cells rows 15..18 in col 0, below the overhang

    # Straight drop for (rot1, col1) rests ON TOP of the overhang, rows 6..9.
    assert env._straight_drop_row(1, 1) == 6

    # Force a decision tick and slide right, under the overhang.
    env.tick_count = 0
    env.gravity_counter = 0
    env.apply_action(RIGHT)
    env.tick()
    assert env.col == 1  # slid under the overhang (destination cells free)
    assert env.row == 15

    # Drive gravity until it locks: one more descent to rows 16..19 (floor).
    lock = None
    for _ in range(GRAVITY_PERIOD * 6):
        lk = env.tick()
        if lk is not None:
            lock = lk
            break
    assert lock is not None
    assert lock["r"] == 1 and lock["c"] == 1
    assert lock["tuck"] is True
    assert env.game_over is False
    assert env.pieces == 1

    # Exact post-lock board: col1 filled at rows 10 (overhang) and 16..19 —
    # the piece stayed where the camera saw it, no teleport to rows 6..9.
    expected = [0] * HEIGHT
    for r in (10, 16, 17, 18, 19):
        expected[r] = 1 << 1
    assert env.rows == expected


def test_above_board_lock_is_game_over():
    # A full column forces a lock whose cells sit above row 0 -> game over with
    # the board left unchanged (bit-identical to the v1 illegal-step outcome).
    env = FrameEnv(seed=0)
    env.piece = I
    env._spawn()
    env.rot = 1  # vertical I in a single column
    env.col = 3
    env.row = -4  # cells rows -4..-1, fully above the board
    for r in range(HEIGHT):
        env.rows[r] = 1 << 3  # col 3 filled at every row
    # Gravity descent immediately collides (row -3 puts a cell at row 0 = filled).
    env.tick_count = 0
    env.gravity_counter = GRAVITY_PERIOD - 1  # next tick triggers gravity
    lk = env.tick()
    assert lk is not None
    assert env.game_over is True
    assert env.pieces == 0  # not counted, mirroring v1 illegal step
    # Board unchanged by the above-board lock.
    assert all(env.rows[r] == (1 << 3) for r in range(HEIGHT))


def test_game_over_freezes_state():
    env = FrameEnv(seed=0)
    env.game_over = True
    rows_before = list(env.rows)
    tc = env.tick_count
    assert env.tick() is None
    assert env.tick_count == tc + 1  # cadence still advances
    assert env.rows == rows_before


def test_determinism():
    def run(seed):
        env = FrameEnv(seed=seed)
        arng = Mulberry32(seed)
        trace = []
        for _ in range(1500):
            if env.is_decision_tick:
                env.apply_action(int(arng.next_float() * 5))
            lk = env.tick()
            trace.append((env.piece, env.rot, env.col, env.row, env.gravity_counter,
                          tuple(env.rows), lk["tick"] if lk else -1))
        return trace

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_v1_consistency_on_non_tuck_prefix():
    # v1-consistency invariant (amended §1): every non-tuck lock transition is
    # bit-identical to v1 engine.step(r, c). Cross-checked by a parallel bare
    # v1 engine over the non-tuck prefix of each random game (boards
    # legitimately diverge after the first physical tuck lock).
    checked = 0
    tucks_seen = 0
    for seed in range(15):
        env = FrameEnv(seed=seed)
        arng = Mulberry32(seed)
        bare = TetrisEngine(seed=seed)
        cross_check = True
        for _ in range(3000):
            if env.is_decision_tick:
                env.apply_action(int(arng.next_float() * 5))
            lk = env.tick()
            if lk is None or not cross_check:
                continue
            if lk["tuck"]:
                tucks_seen += 1
                cross_check = False
                continue
            if bare.game_over:
                cross_check = False
                continue
            bare.step(lk["r"], lk["c"])
            assert bare.rows == env.rows, f"board mismatch seed {seed} @tick {lk['tick']}"
            assert bare.lines == env.lines, f"lines mismatch seed {seed} @tick {lk['tick']}"
            checked += 1
            if bare.game_over:
                # v1's "new piece has no legal placement" rule has no frame-layer
                # equivalent; comparisons after this point are meaningless.
                cross_check = False
    assert checked >= 50  # the invariant was exercised substantially
    assert tucks_seen >= 1  # random play reaches the tuck path
