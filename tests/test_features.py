"""BCTS feature tests on hand-crafted boards (PLAN.md §2, §4).

Board-only features (3-8) are asserted to exact hand-computed values on four
boards, including a multi-well and a multi-hole case. landing_height (1) and
eroded_piece_cells (2) are asserted via hand-constructed placements. A
cross-check confirms the numpy batch path agrees with the pure-Python reference
on random boards.
"""

import numpy as np
import pytest

from tetris.engine import TetrisEngine
from tetris.features import board_features, board_features_batch


def mk(filled_by_row: dict[int, list[int]]) -> list[int]:
    rows = [0] * 20
    for r, cols in filled_by_row.items():
        v = 0
        for c in cols:
            v |= 1 << c
        rows[r] = v
    return rows


# (row_transitions, column_transitions, holes, cumulative_wells, hole_depth,
#  rows_with_holes)
BOARDS = {
    # Empty board.
    "empty": (mk({}), (40, 10, 0, 0, 0, 0)),
    # Two symmetric edge wells (cols 0 and 9) plus one buried hole (18,4).
    "multi_well_hole": (
        mk({17: [1, 4, 8], 18: [1, 8], 19: [1, 4, 8]}),
        (56, 12, 1, 12, 1, 1),
    ),
    # Three holes across two columns; col4 hole buried under 3 cells.
    "multi_hole": (
        mk({15: [4], 16: [4], 17: [4, 6], 19: [4]}),
        (50, 14, 3, 1, 5, 2),
    ),
    # Bumpy surface, single edge well (col 9), no holes.
    "bumpy_well": (
        mk({18: [0, 1], 19: [0, 1, 2, 3, 4, 5, 6, 7, 8]}),
        (40, 10, 0, 1, 0, 0),
    ),
}


@pytest.mark.parametrize("name", list(BOARDS))
def test_board_features_reference(name):
    board, expected = BOARDS[name]
    assert board_features(board) == expected


@pytest.mark.parametrize("name", list(BOARDS))
def test_board_features_batch_matches(name):
    board, expected = BOARDS[name]
    got = tuple(int(x) for x in board_features_batch(np.array([board], dtype=np.uint16))[0])
    assert got == expected


def test_batch_matches_reference_on_random_boards():
    rng = np.random.default_rng(0)
    boards = rng.integers(0, 1024, size=(200, 20), dtype=np.uint16)
    batch = board_features_batch(boards)
    for i in range(boards.shape[0]):
        ref = board_features([int(x) for x in boards[i]])
        assert tuple(int(x) for x in batch[i]) == ref, f"mismatch at board {i}"


def _step_on(rows, current, rotation, column):
    e = TetrisEngine(seed=1)
    e.rows = list(rows)
    e.current = current
    info = e.step(rotation, column)
    return e, info


def test_landing_height_o_on_empty():
    # O (index 1) dropped flat on the floor: cells rest on rows 18,19.
    # rows-from-bottom = {2, 1}; mean = 1.5.
    _, info = _step_on([0] * 20, current=1, rotation=0, column=0)
    assert info.landing_height == 1.5
    assert info.eroded_piece_cells == 0
    assert info.lines_cleared == 0


def test_single_line_clear_and_eroded():
    # Bottom row filled except cols 0,1; O fills them -> row 19 clears.
    rows = [0] * 20
    rows[19] = 0x3FC  # cols 2..9
    e, info = _step_on(rows, current=1, rotation=0, column=0)
    assert info.lines_cleared == 1
    # Two of the O's four cells were in the cleared line.
    assert info.eroded_piece_cells == 2
    assert info.landing_height == 1.5
    # After the clear only the top O cells remain, shifted to the floor.
    assert e.rows[19] == 0b11
    assert sum(e.rows) == 0b11


def test_tetris_four_line_clear():
    # Cols 1..9 filled for the bottom 4 rows; vertical I fills the col-0 well.
    rows = [0] * 20
    for r in (16, 17, 18, 19):
        rows[r] = 0x3FE  # cols 1..9
    e, info = _step_on(rows, current=0, rotation=1, column=0)
    assert info.lines_cleared == 4
    # All four I cells were in cleared lines: 4 lines * 4 cells.
    assert info.eroded_piece_cells == 16
    # I vertical spans rows 16..19 -> rows-from-bottom {4,1}; mean = 2.5.
    assert info.landing_height == 2.5
    assert sum(e.rows) == 0  # board cleared


def test_full_feature_vector_shape_and_order():
    # step returns the full 8-vector: [landing, eroded, + 6 board features].
    rows = [0] * 20
    rows[19] = 0x3FC
    _, info = _step_on(rows, current=1, rotation=0, column=0)
    assert len(info.features) == 8
    assert info.features[0] == 1.5  # landing_height
    assert info.features[1] == 2  # eroded_piece_cells
