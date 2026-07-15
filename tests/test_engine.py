"""Engine drop/clear, enumeration, and determinism tests (PLAN.md §2, §4)."""

import numpy as np

from tetris.agents import RandomAgent, dellacherie_agent
from tetris.engine import PIECES, TetrisEngine
from tetris.features import board_features_batch


def test_drop_on_empty_board_rests_on_floor():
    e = TetrisEngine(seed=1)
    e.rows = [0] * 20
    e.current = 1  # O
    e.step(0, 0)
    assert e.rows[18] == 0b11
    assert e.rows[19] == 0b11
    assert e.lines == 0


def test_column_range_matches_width():
    # Legal columns for a rotation span [0, 10 - width]; on an empty board every
    # column is legal, so counts equal 10 - width + 1 per rotation.
    e = TetrisEngine(seed=1)
    e.rows = [0] * 20
    for piece in range(7):
        e.current = piece
        placements = e.legal_placements()
        expected = sum(rot.n_placements for rot in PIECES[piece])
        assert len(placements) == expected


def test_placements_in_enumeration_order():
    e = TetrisEngine(seed=1)
    e.rows = [0] * 20
    e.current = 2  # T, 4 rotations
    placements = e.legal_placements()
    assert placements == sorted(placements)  # (rotation asc, column asc)


def test_illegal_placement_ends_game():
    e = TetrisEngine(seed=1)
    # Fill everything except the top row so the piece can't rest with row >= 0.
    for r in range(1, 20):
        e.rows[r] = 0x3FF
    e.current = 1  # O needs two rows; only row 0 is free -> illegal
    info = e.step(0, 0)
    assert info.game_over is True


def test_candidate_features_matches_step_features():
    # The 8-vector from candidate_features for a placement must equal the vector
    # step() reports for the same placement.
    e = TetrisEngine(seed=5)
    e.rows[19] = 0x0FF  # cols 0..7 filled on the floor
    e.current = 3  # S
    placements, feats, _ = e.candidate_features()
    for idx, (rot, col) in enumerate(placements):
        probe = e.clone()
        info = probe.step(rot, col)
        assert tuple(float(x) for x in feats[idx]) == tuple(float(x) for x in info.features)


def test_afterstates_feed_board_features():
    e = TetrisEngine(seed=9)
    e.current = 2
    placements, feats, afters = e.candidate_features()
    board6 = board_features_batch(np.array(afters, dtype=np.uint16))
    assert np.array_equal(feats[:, 2:], board6.astype(np.float64))


def _trajectory(seed, agent_seed, n):
    e = TetrisEngine(seed=seed)
    agent = RandomAgent(seed=agent_seed)
    trace = []
    steps = 0
    while not e.game_over and steps < n:
        move = agent.act(e)
        if move is None:
            break
        info = e.step(*move)
        trace.append((move, info.lines_cleared, tuple(e.rows)))
        steps += 1
    return trace, e.lines, e.pieces


def test_determinism_1000_steps():
    a = _trajectory(seed=42, agent_seed=123, n=1000)
    b = _trajectory(seed=42, agent_seed=123, n=1000)
    assert a == b
    # A different seed should generally diverge.
    c = _trajectory(seed=43, agent_seed=123, n=1000)
    assert a != c


def test_dellacherie_clears_lines_short_game():
    # Sanity: the hand-weighted agent clears lines efficiently over a short cap.
    e = TetrisEngine(seed=7)
    agent = dellacherie_agent()
    while not e.game_over and e.pieces < 500:
        move = agent.act(e)
        if move is None:
            break
        e.step(*move)
    assert e.lines >= 100  # ~0.4 lines/piece expected; conservative floor
