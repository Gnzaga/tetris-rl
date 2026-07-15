"""BCTS board features on bitboards (PLAN.md §2, features 3-8).

Six of the eight BCTS features depend only on the post-clear board:
row_transitions, column_transitions, holes, cumulative_wells, hole_depth,
rows_with_holes. The remaining two (landing_height, eroded_piece_cells) depend
on the placement itself and are produced by the engine.

A board is 20 rows of 10-bit ints (bit c set = column c filled), row 0 top,
row 19 bottom. Everything here is bit-parallel: `board_features_batch` scores
many candidate boards at once with numpy for the hot decision path;
`board_features` is a pure-Python reference for a single board.
"""

from __future__ import annotations

import numpy as np

WIDTH = 10
HEIGHT = 20
FULL_ROW = (1 << WIDTH) - 1  # 0x3FF

# --- Lookup tables over all 10-bit row values -------------------------------

_TABLE_SIZE = 1 << WIDTH  # 1024

# Horizontal transitions per row with both side walls counted as filled
# (PLAN.md §2 feature 3). Sequence = [wall, b0..b9, wall]; count adjacent
# differences. Packed as bit0=left wall, bits1..10=cells, bit11=right wall.
_ROW_TRANS = np.empty(_TABLE_SIZE, dtype=np.int64)
_POPCOUNT = np.empty(_TABLE_SIZE, dtype=np.int64)
for _v in range(_TABLE_SIZE):
    _a = (_v << 1) | 0x801
    _ROW_TRANS[_v] = ((_a ^ (_a >> 1)) & 0x7FF).bit_count()
    _POPCOUNT[_v] = _v.bit_count()

_BITPOS = np.arange(WIDTH, dtype=np.uint16)


def _bits(x: np.ndarray) -> np.ndarray:
    """(P,) row ints -> (P, WIDTH) array of 0/1 per column."""
    return ((x[:, None] >> _BITPOS) & 1).astype(np.int64)


def board_features_batch(rows: np.ndarray) -> np.ndarray:
    """Compute the 6 board-only BCTS features for many boards at once.

    Args:
        rows: (P, 20) array of 10-bit row ints (uint16-compatible).

    Returns:
        (P, 6) int64 array:
        [row_transitions, column_transitions, holes, cumulative_wells,
         hole_depth, rows_with_holes].
    """
    rows = np.asarray(rows, dtype=np.uint16)
    if rows.ndim != 2 or rows.shape[1] != HEIGHT:
        raise ValueError(f"expected (P, {HEIGHT}) rows, got {rows.shape}")
    P = rows.shape[0]

    # (3) row transitions: table lookup summed over all 20 rows.
    row_trans = _ROW_TRANS[rows].sum(axis=1)

    # (4) column transitions: top border empty, floor filled.
    col_trans = _POPCOUNT[rows[:, 0]].copy()
    for r in range(HEIGHT - 1):
        col_trans += _POPCOUNT[rows[:, r] ^ rows[:, r + 1]]
    col_trans += _POPCOUNT[rows[:, HEIGHT - 1] ^ FULL_ROW]

    # (5)(6)(7)(8): single top-down scan maintaining per-column state.
    above_or = np.zeros(P, dtype=np.uint16)
    col_filled = np.zeros((P, WIDTH), dtype=np.int64)
    run_len = np.zeros((P, WIDTH), dtype=np.int64)
    holes = np.zeros(P, dtype=np.int64)
    hole_depth = np.zeros(P, dtype=np.int64)
    rows_with_holes = np.zeros(P, dtype=np.int64)
    cum_wells = np.zeros(P, dtype=np.int64)

    for r in range(HEIGHT):
        row = rows[:, r]
        rb = _bits(row)
        ab = _bits(above_or)
        hole_cells = ab & (1 - rb)  # empty now, filled somewhere above
        n_holes = hole_cells.sum(axis=1)
        holes += n_holes
        rows_with_holes += (n_holes > 0).astype(np.int64)
        # hole_depth counts ALL filled cells above each hole in its column (not
        # just the contiguous run) — pinned by test, mirrored in JS engine.
        hole_depth += (hole_cells * col_filled).sum(axis=1)
        col_filled += rb
        above_or = above_or | row

        # (6) cumulative wells: empty cell whose left & right neighbors are
        # filled (walls filled); triangular sum over maximal vertical runs.
        left = ((row << np.uint16(1)) | np.uint16(1)) & np.uint16(FULL_ROW)
        right = ((row >> np.uint16(1)) | np.uint16(1 << (WIDTH - 1))) & np.uint16(FULL_ROW)
        well = (~row) & left & right & np.uint16(FULL_ROW)
        wb = _bits(well)
        run_len = np.where(wb == 1, run_len + 1, 0)
        cum_wells += run_len.sum(axis=1)

    return np.stack(
        [row_trans, col_trans, holes, cum_wells, hole_depth, rows_with_holes],
        axis=1,
    )


def board_features(rows) -> tuple[int, int, int, int, int, int]:
    """Pure-Python reference for the 6 board features of a single board.

    Returns (row_transitions, column_transitions, holes, cumulative_wells,
    hole_depth, rows_with_holes).
    """
    rows = list(rows)
    if len(rows) != HEIGHT:
        raise ValueError(f"expected {HEIGHT} rows, got {len(rows)}")

    row_trans = 0
    for v in rows:
        a = (v << 1) | 0x801
        row_trans += ((a ^ (a >> 1)) & 0x7FF).bit_count()

    col_trans = (rows[0] & FULL_ROW).bit_count()
    for r in range(HEIGHT - 1):
        col_trans += ((rows[r] ^ rows[r + 1]) & FULL_ROW).bit_count()
    col_trans += ((rows[HEIGHT - 1] ^ FULL_ROW) & FULL_ROW).bit_count()

    holes = 0
    hole_depth = 0
    rows_with_holes = 0
    col_filled = [0] * WIDTH
    above_or = 0
    for r in range(HEIGHT):
        row = rows[r]
        hole_bits = above_or & ~row & FULL_ROW
        if hole_bits:
            rows_with_holes += 1
            m = hole_bits
            while m:
                c = (m & -m).bit_length() - 1
                holes += 1
                # hole_depth counts ALL filled cells above each hole in its
                # column (not just the contiguous run) — pinned by test,
                # mirrored in JS engine.
                hole_depth += col_filled[c]
                m &= m - 1
        m = row & FULL_ROW
        while m:
            c = (m & -m).bit_length() - 1
            col_filled[c] += 1
            m &= m - 1
        above_or |= row

    cum_wells = 0
    run_len = [0] * WIDTH
    for r in range(HEIGHT):
        row = rows[r]
        left = ((row << 1) | 1) & FULL_ROW
        right = ((row >> 1) | (1 << (WIDTH - 1))) & FULL_ROW
        well = ~row & left & right & FULL_ROW
        for c in range(WIDTH):
            if well & (1 << c):
                run_len[c] += 1
                cum_wells += run_len[c]
            else:
                run_len[c] = 0

    return (row_trans, col_trans, holes, cum_wells, hole_depth, rows_with_holes)
