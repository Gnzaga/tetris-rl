"""96x96 grayscale observation rasterizer (PLAN2.md §1, Phase C; perception
amendment).

Renders the "camera" the pixel agent sees: an integer-aligned, anti-aliasing-free
96x96 uint8 grayscale image of the frame-layer state. Filled stack cells, the
next-piece preview, and the board border render 255; the ACTIVE piece renders
gray 128 (§1 perception amendment — a real Tetris screen draws the falling piece
in its own color, so the camera may see it distinctly); empty is 0.

Frozen pixel layout (identical here and in demo/js/render_obs.js; changing any
constant breaks the bit-exact CRC32 fixtures):

* Canvas: 96x96, row-major, ``buf[y * 96 + x]`` (y down, x right).
* Board: 10 columns x 20 rows at 4 px/cell (40x80 px). Its top-left interior
  pixel is (x=8, y=8), so the interior occupies x in [8, 47], y in [8, 87].
  Board cell (row r, col c) fills the 4x4 rect with top-left (8 + 4c, 8 + 4r).
* Border: a 1 px white ring immediately AROUND the board — the perimeter of the
  rectangle whose inclusive corners are (7, 7) and (48, 88) (i.e. the ring one
  pixel outside the 40x80 interior on every side). Its edges live at x=7, x=48,
  y=7, y=88.
* Preview: the NEXT piece (queue[0]) drawn at its rotation-0 bounding box,
  top-left aligned in a 20x20 region with top-left (x=56, y=8) — interior
  x in [56, 75], y in [8, 27] — at 4 px/cell, NO border. Preview cell (r, c)
  fills the 4x4 rect with top-left (56 + 4c, 8 + 4r).

Active-piece cells at negative board rows (above the visible board) are simply
not drawn.
"""

from __future__ import annotations

import numpy as np

from tetris.engine import PIECES
from tetris.features import HEIGHT, WIDTH

# --- Frozen pixel constants (see module docstring) --------------------------
OBS_SIZE = 96
CELL = 4

BOARD_X = 8  # interior top-left x (column 0)
BOARD_Y = 8  # interior top-left y (row 0)

# Border ring: perimeter of the rectangle with inclusive corners
# (BORDER_X0, BORDER_Y0) .. (BORDER_X1, BORDER_Y1). One pixel outside the board
# interior (which is 40x80 starting at (8, 8)) on every side.
BORDER_X0 = BOARD_X - 1               # 7
BORDER_Y0 = BOARD_Y - 1               # 7
BORDER_X1 = BOARD_X + WIDTH * CELL    # 48  (interior ends at x=47)
BORDER_Y1 = BOARD_Y + HEIGHT * CELL   # 88  (interior ends at y=87)

PREVIEW_X = 56
PREVIEW_Y = 8

FILLED = 255
ACTIVE = 128  # active-piece gray (§1 perception amendment)


def _fill_cell(buf: np.ndarray, x0: int, y0: int, value: int = FILLED) -> None:
    """Set the 4x4 block whose top-left is (x0, y0) to ``value`` (in-place)."""
    buf[y0:y0 + CELL, x0:x0 + CELL] = value


def render_obs(rows, piece: int, rot: int, col: int, row: int,
               next_piece: int) -> np.ndarray:
    """Rasterize one observation to a (96, 96) uint8 array (see module docstring).

    ``rows``      : 20 row ints (bit c set = column c filled) — the locked stack.
    ``piece/rot/col/row`` : the active piece pose (top-left board anchor).
    ``next_piece``: piece id drawn in the preview at its rotation-0 bbox.
    """
    buf = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.uint8)

    # Border ring (perimeter of (7,7)..(48,88)).
    buf[BORDER_Y0, BORDER_X0:BORDER_X1 + 1] = FILLED
    buf[BORDER_Y1, BORDER_X0:BORDER_X1 + 1] = FILLED
    buf[BORDER_Y0:BORDER_Y1 + 1, BORDER_X0] = FILLED
    buf[BORDER_Y0:BORDER_Y1 + 1, BORDER_X1] = FILLED

    # Locked stack cells.
    for r in range(HEIGHT):
        bits = rows[r]
        if not bits:
            continue
        y0 = BOARD_Y + r * CELL
        for c in range(WIDTH):
            if (bits >> c) & 1:
                _fill_cell(buf, BOARD_X + c * CELL, y0)

    # Active piece renders gray ACTIVE=128 (cells above the board, row+ro < 0,
    # are not drawn). Collision rules guarantee no overlap with the stack.
    for ro, co in PIECES[piece][rot].cells:
        rr = row + ro
        cc = col + co
        if 0 <= rr < HEIGHT and 0 <= cc < WIDTH:
            _fill_cell(buf, BOARD_X + cc * CELL, BOARD_Y + rr * CELL, ACTIVE)

    # Next-piece preview at its rotation-0 bbox, top-left aligned, no border.
    for ro, co in PIECES[next_piece][0].cells:
        _fill_cell(buf, PREVIEW_X + co * CELL, PREVIEW_Y + ro * CELL)

    return buf


def render_env(env) -> np.ndarray:
    """Render the current observation for a :class:`~tetris.frame_env.FrameEnv`.

    The preview shows ``env.queue[0]`` (the NEXT piece). Requires at least one
    queued piece, which the frame env always maintains (preview >= 1)."""
    return render_obs(env.rows, env.piece, env.rot, env.col, env.row,
                      env.queue[0])
