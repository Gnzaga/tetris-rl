"""Frame-layer unit tests (PLAN2.md §3).

Covers gravity / lock / spawn semantics on hand-built scenarios, the rotation
clamp and its failure cases, above-board locks ending the game, frame-layer
determinism, v1-consistency (the lock sequence replayed through a bare v1 engine
reproduces identical boards), the straight-drop invariant, and a constructed
tuck (the one reachable exception to frame_row == drop_row).
"""

import pytest

from tetris.engine import PIECES, TetrisEngine
from tetris.features import HEIGHT, WIDTH
from tetris.frame_env import (
    DECISION_PERIOD,
    GRAVITY_PERIOD,
    LEFT,
    NOOP,
    RIGHT,
    ROT_CCW,
    ROT_CW,
    FrameEnv,
)
from tetris.rng import Mulberry32

# Piece indices (shared/pieces.json order): I O T S Z J L
I, O, T, S, Z, J, L = range(7)


def drive(env, actions, ticks):
    """Advance `ticks` ticks, applying actions[i] on decision tick i (or NOOP).

    `actions` maps decision-index -> action; missing indices are NOOP. Returns
    the list of lock events emitted.
    """
    locks = []
    dec_i = 0
    for _ in range(ticks):
        if env.is_decision_tick:
            a = actions.get(dec_i, NOOP) if isinstance(actions, dict) else NOOP
            env.apply_action(a)
            dec_i += 1
        lk = env.tick()
        if lk is not None:
            locks.append(lk)
    return locks


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
    env.engine.rows[5] = 1 << 6  # col6 filled at row5
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
    # Horizontal I at (rot0, col clamped, row 16) covers row 16 cols col..col+3.
    env.engine.rows[16] = (1 << WIDTH) - 1  # entire row 16 filled
    env.tick_count = 0
    env.apply_action(ROT_CW)
    env.tick()
    assert env.rot == 1  # rotation failed, unchanged
    assert env.col == 5


def test_lock_on_floor_and_v1_consistency():
    # Drop an I piece straight to the floor with no actions; it must lock at the
    # v1 straight-drop position and the board must match a bare engine.step.
    env = FrameEnv(seed=0)
    env.piece = I
    env.engine.current = I  # keep engine's locking piece in sync with the frame
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
    assert env.engine.rows[HEIGHT - 1] == 0b0000_1111_000  # cols 3,4,5,6

    # v1-consistency: replay the derived placement through a bare engine.
    bare = TetrisEngine(seed=0)
    bare.current = I
    bare.step(lock["r"], lock["c"])
    assert bare.rows == env.engine.rows


def test_straight_drop_invariant_no_tuck():
    # Over a full random game the engine drop row never exceeds the frame row.
    env = FrameEnv(seed=3)
    arng = Mulberry32(3)
    for _ in range(3000):
        if env.is_decision_tick:
            env.apply_action(int(arng.next_float() * 5))
        # tick() asserts drop <= row internally; reaching here means it held.
        env.tick()


def test_constructed_tuck():
    # A vertical I piece slides under an overhang, resting deeper than any v1
    # straight drop of (rot=1, col=1) could reach. The lock still defers to
    # engine.step so the board stays v1-consistent, and the tuck flag is set.
    env = FrameEnv(seed=0)
    env.piece = I
    env.engine.current = I  # sync engine's locking piece with the frame
    env._spawn()
    env.engine.rows = [0] * HEIGHT
    env.engine.rows[10] = 1 << 1  # overhang: col1 filled at row 10, empty below
    env.rot = 1  # vertical I, width 1
    env.col = 0  # open column
    env.row = 15  # cells rows 15..18 in col 0, below the overhang

    # Engine straight-drop for (rot1, col1) rests ON TOP of the row-10 block, at
    # rows 6..9 — far above where the tucked piece will actually rest.
    drop_col1 = env.engine._drop_row(PIECES[I][1], 1, env.engine._col_top())
    assert drop_col1 == 6

    # Force a decision tick and slide right, under the overhang.
    env.tick_count = 0
    env.gravity_counter = 0
    env.apply_action(RIGHT)
    env.tick()
    assert env.col == 1  # slid under the overhang (destination cells free)
    assert env.row == 15

    # Drive gravity until it locks.
    lock = None
    for _ in range(GRAVITY_PERIOD * 6):
        lk = env.tick()
        if lk is not None:
            lock = lk
            break
    assert lock is not None
    assert lock["r"] == 1 and lock["c"] == 1
    assert lock["tuck"] is True  # rested deeper than the straight drop
    assert env.game_over is False  # the derived placement is legal (on-board)

    # v1-consistency holds anyway: the board equals a bare straight drop of the
    # derived placement onto the same pre-lock board (piece lands on the block,
    # NOT at the tucked-deep visual position).
    bare = TetrisEngine(seed=0)
    bare.rows = [0] * HEIGHT
    bare.rows[10] = 1 << 1
    bare.current = I
    bare.step(lock["r"], lock["c"])
    assert bare.rows == env.engine.rows
    # The engine locked the I at the straight-drop rows 6..9, not rows 16..19.
    assert all((env.engine.rows[r] >> 1) & 1 for r in range(6, 11))


def test_above_board_lock_is_game_over():
    # A near-full column forces a lock whose piece sits above row 0 -> the v1
    # engine reports the placement illegal and the game ends.
    env = FrameEnv(seed=0)
    env.piece = I
    env.engine.current = I  # sync engine's locking piece with the frame
    env._spawn()
    # Fill col 3 entirely from row 0 down, so a vertical I above it cannot fit.
    env.rot = 1  # vertical I in a single column
    env.col = 3
    env.row = -4  # cells rows -4..-1, fully above the board
    for r in range(HEIGHT):
        env.engine.rows[r] = 1 << 3  # col 3 filled at every row
    # Gravity descent immediately collides (row -3 puts a cell at row 0 = filled).
    env.tick_count = 0
    env.gravity_counter = GRAVITY_PERIOD - 1  # next tick triggers gravity
    lk = env.tick()
    assert lk is not None
    assert env.game_over is True
    assert env.engine.game_over is True
    # Board unchanged by the illegal lock.
    assert all(env.engine.rows[r] == (1 << 3) for r in range(HEIGHT))


def test_game_over_freezes_state():
    env = FrameEnv(seed=0)
    env.game_over = True
    rows_before = list(env.engine.rows)
    tc = env.tick_count
    assert env.tick() is None
    assert env.tick_count == tc + 1  # cadence still advances
    assert env.engine.rows == rows_before


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
                          tuple(env.engine.rows), lk["tick"] if lk else -1))
        return trace

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_frame_lock_sequence_matches_bare_engine():
    # v1-consistency at scale: replay every derived lock placement from a random
    # frame game through a bare v1 engine and assert identical final board/lines.
    for seed in range(15):
        env = FrameEnv(seed=seed)
        arng = Mulberry32(seed)
        bare = TetrisEngine(seed=seed)
        for _ in range(3000):
            if env.is_decision_tick:
                env.apply_action(int(arng.next_float() * 5))
            lk = env.tick()
            if lk is not None and not bare.game_over:
                bare.step(lk["r"], lk["c"])
        assert bare.rows == env.engine.rows, f"board mismatch seed {seed}"
        assert bare.lines == env.engine.lines, f"lines mismatch seed {seed}"
