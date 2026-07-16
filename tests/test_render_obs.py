"""Observation-renderer tests (PLAN2.md §1/§5, Phase C).

Covers the frozen 96x96 pixel layout on hand-built states (border ring at the
exact coordinates, board-cell -> 4x4 rect mapping, active piece drawn / above-
board cells omitted, preview at its region), value invariants ({0,255} only),
and that the recorded obs_v2.json CRC32 fixtures reproduce from Python (the same
bytes the JS suite asserts bit-exactly).
"""

import json
import zlib
from pathlib import Path

import numpy as np

from tetris.frame_env import DECISION_PERIOD, FrameEnv
from tetris.render_obs import (
    BOARD_X,
    BOARD_Y,
    BORDER_X0,
    BORDER_X1,
    BORDER_Y0,
    BORDER_Y1,
    CELL,
    OBS_SIZE,
    PREVIEW_X,
    PREVIEW_Y,
    render_env,
    render_obs,
)
from tetris.rng import Mulberry32

# Piece indices (shared/pieces.json order): I O T S Z J L
I, O, T, S, Z, J, L = range(7)
EMPTY = [0] * 20


def test_shape_dtype_and_binary_values():
    obs = render_obs(EMPTY, I, 0, 3, -1, O)
    assert obs.shape == (OBS_SIZE, OBS_SIZE)
    assert obs.dtype == np.uint8
    assert set(np.unique(obs).tolist()) <= {0, 255}


def test_border_ring_exact_coords():
    # Empty board, active piece hidden above (row very negative), preview = O in
    # the preview region only. Then every board-interior pixel is 0 and the only
    # set pixels outside the preview region are the border ring.
    obs = render_obs(EMPTY, I, 0, 3, -50, O)
    # Ring edges are 255 along the full inclusive spans.
    assert np.all(obs[BORDER_Y0, BORDER_X0:BORDER_X1 + 1] == 255)
    assert np.all(obs[BORDER_Y1, BORDER_X0:BORDER_X1 + 1] == 255)
    assert np.all(obs[BORDER_Y0:BORDER_Y1 + 1, BORDER_X0] == 255)
    assert np.all(obs[BORDER_Y0:BORDER_Y1 + 1, BORDER_X1] == 255)
    # Just outside the ring is empty (top-left corner neighborhood).
    assert obs[BORDER_Y0 - 1, BORDER_X0] == 0
    assert obs[BORDER_Y0, BORDER_X0 - 1] == 0
    # Board interior is empty.
    assert np.all(obs[BOARD_Y:BOARD_Y + 80, BOARD_X:BOARD_X + 40] == 0)


def test_board_cell_maps_to_exact_4x4_rect():
    # A single filled stack cell at (row 5, col 3) -> 4x4 rect at (8+12, 8+20).
    rows = [0] * 20
    rows[5] = 1 << 3
    obs = render_obs(rows, I, 0, 3, -50, O)
    x0 = BOARD_X + 3 * CELL
    y0 = BOARD_Y + 5 * CELL
    assert np.all(obs[y0:y0 + CELL, x0:x0 + CELL] == 255)
    # The neighboring cell to the left/above is empty.
    assert np.all(obs[y0:y0 + CELL, x0 - CELL:x0] == 0)
    assert np.all(obs[y0 - CELL:y0, x0:x0 + CELL] == 0)


def test_active_piece_drawn_and_above_board_omitted():
    # O piece straddling the top edge: bbox rows {row, row+1}. With row = -1 the
    # top cells (row -1) are above the board (omitted) and the bottom cells
    # (row 0) are drawn.
    obs = render_obs(EMPTY, O, 0, 4, -1, I)
    # Bottom row of the O (board row 0) at cols 4,5 is drawn.
    for c in (4, 5):
        x0 = BOARD_X + c * CELL
        assert np.all(obs[BOARD_Y:BOARD_Y + CELL, x0:x0 + CELL] == 255)
    # Nothing is drawn above the board border (rows above BORDER contain only the
    # ring / preview, never active-piece cells).
    # Fully above the board => no active cells anywhere in the interior.
    obs_above = render_obs(EMPTY, O, 0, 4, -5, I)
    assert np.all(obs_above[BOARD_Y:BOARD_Y + 80, BOARD_X:BOARD_X + 40] == 0)


def test_preview_region_and_no_border():
    # O preview -> a 2x2 cell block (8x8 px) at the preview top-left, no ring.
    obs = render_obs(EMPTY, I, 0, 3, -50, O)
    assert np.all(obs[PREVIEW_Y:PREVIEW_Y + 2 * CELL, PREVIEW_X:PREVIEW_X + 2 * CELL] == 255)
    # No border pixels around the preview (immediately left of it is empty).
    assert np.all(obs[PREVIEW_Y:PREVIEW_Y + 2 * CELL, PREVIEW_X - 1] == 0)


def test_active_piece_not_distinct_from_stack():
    # A stack cell and an active-piece cell landing on the same board cell render
    # identically (both 255) — the camera cannot tell them apart.
    rows = [0] * 20
    rows[10] = 1 << 2
    a = render_obs(rows, I, 0, 3, -50, O)           # cell only in stack
    b = render_obs([0] * 20, I, 1, 2, 10, O)         # vertical I overlapping col 2
    x0, y0 = BOARD_X + 2 * CELL, BOARD_Y + 10 * CELL
    assert np.all(a[y0:y0 + CELL, x0:x0 + CELL] == 255)
    assert np.all(b[y0:y0 + CELL, x0:x0 + CELL] == 255)


def test_obs_fixtures_reproduce():
    # The recorded obs_v2.json CRC32s reproduce from a fresh Python render — the
    # exact bytes the JS suite asserts bit-exactly.
    path = Path(__file__).resolve().parent.parent / "shared" / "fixtures" / "obs_v2.json"
    fx = json.loads(path.read_text())
    num_actions = fx["num_actions"]
    for entry in fx["fixtures"]:
        seed = entry["seed"]
        env = FrameEnv(seed=seed)
        rng = Mulberry32(seed)
        got = []
        while len(got) < len(entry["crcs"]) and not env.game_over:
            if env.tick_count % DECISION_PERIOD == 0:
                got.append(zlib.crc32(render_env(env).tobytes()) & 0xFFFFFFFF)
                env.apply_action(int(rng.next_float() * num_actions))
            env.tick()
        assert got == entry["crcs"], f"obs CRC mismatch for seed {seed}"
