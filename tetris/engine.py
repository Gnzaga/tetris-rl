"""Bitboard Tetris simulator (PLAN.md §2).

Board: 10 columns x 20 rows, row 0 top / row 19 bottom, each row a 10-bit int
(bit c set = column c filled). The engine's only transition is the atomic
`step(rotation, column)`; there is no per-frame gravity — a placement drops
straight to its resting position. The RNG-driven 7-bag feeds the current piece
and a 5-piece preview queue.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from .features import FULL_ROW, HEIGHT, WIDTH, board_features_batch
from .rng import Mulberry32, SevenBag

_PIECES_PATH = Path(__file__).resolve().parent.parent / "shared" / "pieces.json"

# Scoring / reward tables (PLAN.md §2).
CLEAR_POINTS = (0, 1, 3, 5, 8)


class Rotation:
    """A single rotation state: cells plus precomputed drop/geometry data."""

    __slots__ = ("width", "height", "cells", "bottom", "n_placements")

    def __init__(self, width: int, height: int, cells: list[tuple[int, int]]):
        self.width = width
        self.height = height
        self.cells = tuple((r, c) for r, c in cells)
        # bottom[pc] = lowest (max) row offset among cells in piece-column pc.
        bottom = [0] * width
        for pc in range(width):
            bottom[pc] = max(r for r, c in cells if c == pc)
        self.bottom = tuple(bottom)
        self.n_placements = WIDTH - width + 1


def _load_pieces() -> list[list[Rotation]]:
    with open(_PIECES_PATH) as f:
        data = json.load(f)
    pieces = []
    for piece in data["pieces"]:
        rots = [
            Rotation(rot["width"], rot["height"], [tuple(c) for c in rot["cells"]])
            for rot in piece["rotations"]
        ]
        pieces.append(rots)
    return pieces


PIECES: list[list[Rotation]] = _load_pieces()
NUM_PIECES = len(PIECES)


class StepInfo:
    """Per-placement outcome returned by `step` (PLAN.md §2)."""

    __slots__ = (
        "lines_cleared",
        "game_over",
        "landing_height",
        "eroded_piece_cells",
        "features",
    )

    def __init__(self, lines_cleared, game_over, landing_height, eroded, features):
        self.lines_cleared = lines_cleared
        self.game_over = game_over
        self.landing_height = landing_height
        self.eroded_piece_cells = eroded
        self.features = features  # full 8-vector for the applied placement


class TetrisEngine:
    """Deterministic bitboard Tetris engine driven by `step(rotation, column)`."""

    def __init__(self, seed: int, preview: int = 5):
        self.rng = Mulberry32(seed)
        self.bag = SevenBag(self.rng)
        self.preview = preview
        self.rows: list[int] = [0] * HEIGHT
        self.queue: deque[int] = deque()
        self._fill_queue()
        self.current: int = self.queue.popleft()
        self._fill_queue()
        self.lines = 0
        self.pieces = 0
        self.game_over = False

    # -- piece supply --------------------------------------------------------

    def _fill_queue(self) -> None:
        while len(self.queue) < self.preview:
            self.queue.append(self.bag.next_piece())

    def preview_pieces(self) -> list[int]:
        """The next `preview` piece indices (does not include `current`)."""
        return list(self.queue)[: self.preview]

    # -- geometry ------------------------------------------------------------

    def _col_top(self) -> list[int]:
        """Topmost filled row per column; HEIGHT (=20) if the column is empty."""
        top = [HEIGHT] * WIDTH
        seen = 0
        for r in range(HEIGHT):
            nb = self.rows[r] & ~seen & FULL_ROW
            if nb:
                seen |= nb
                m = nb
                while m:
                    c = (m & -m).bit_length() - 1
                    top[c] = r
                    m &= m - 1
                if seen == FULL_ROW:
                    break
        return top

    def _drop_row(self, rot: Rotation, col: int, col_top: list[int]) -> int:
        """Resting top-row of the bounding box (may be negative => illegal)."""
        t = HEIGHT
        for pc in range(rot.width):
            v = col_top[col + pc] - 1 - rot.bottom[pc]
            if v < t:
                t = v
        return t

    def legal_placements(self, piece: int | None = None) -> list[tuple[int, int]]:
        """All legal (rotation, column) placements for `piece` (default current),
        in enumeration order: rotation ascending, then column ascending."""
        if piece is None:
            piece = self.current
        col_top = self._col_top()
        out = []
        for rot_idx, rot in enumerate(PIECES[piece]):
            for col in range(rot.n_placements):
                if self._drop_row(rot, col, col_top) >= 0:
                    out.append((rot_idx, col))
        return out

    # -- placement simulation ------------------------------------------------

    def _apply(self, rot: Rotation, col: int, top_row: int):
        """Lock the piece at `top_row` and clear lines.

        Returns (post_rows, lines_cleared, landing_height, eroded_piece_cells).
        Does not mutate engine state.
        """
        pre = self.rows.copy()
        for ro, co in rot.cells:
            pre[top_row + ro] |= 1 << (col + co)

        full_rows = set()
        for r in range(top_row, top_row + rot.height):
            if pre[r] == FULL_ROW:
                full_rows.add(r)
        lines = len(full_rows)

        eroded = 0
        if lines:
            piece_cells_cleared = sum(
                1 for ro, co in rot.cells if (top_row + ro) in full_rows
            )
            eroded = lines * piece_cells_cleared

        # landing height: mean of rows-from-bottom of the piece's lowest and
        # highest cells (pre-clear), rows-from-bottom = 20 - row.
        max_r = top_row + rot.height - 1
        min_r = top_row
        landing_height = ((HEIGHT - max_r) + (HEIGHT - min_r)) / 2.0

        if lines:
            kept = [v for v in pre if v != FULL_ROW]
            post = [0] * lines + kept
        else:
            post = pre
        return post, lines, landing_height, eroded

    def candidate_features(self):
        """Enumerate legal placements for the current piece and return their
        afterstates and full 8-feature vectors.

        Returns (placements, feats, afterstates) where:
          placements   : list of (rotation, column) in enumeration order
          feats         : (P, 8) float array — the 8 BCTS features per placement
          afterstates   : list of post-clear row lists (length 20 each)
        """
        col_top = self._col_top()
        placements: list[tuple[int, int]] = []
        afters: list[list[int]] = []
        landing: list[float] = []
        eroded: list[int] = []
        for rot_idx, rot in enumerate(PIECES[self.current]):
            for col in range(rot.n_placements):
                top_row = self._drop_row(rot, col, col_top)
                if top_row < 0:
                    continue
                post, lines, lh, ero = self._apply(rot, col, top_row)
                placements.append((rot_idx, col))
                afters.append(post)
                landing.append(lh)
                eroded.append(ero)

        if not placements:
            return [], np.empty((0, 8), dtype=np.float64), []

        rows_arr = np.array(afters, dtype=np.uint16)
        board6 = board_features_batch(rows_arr).astype(np.float64)
        feats = np.empty((len(placements), 8), dtype=np.float64)
        feats[:, 0] = landing
        feats[:, 1] = eroded
        feats[:, 2:] = board6
        return placements, feats, afters

    # -- transition ----------------------------------------------------------

    def step(self, rotation: int, column: int) -> StepInfo:
        """Apply a placement atomically, advance the piece, update counters."""
        if self.game_over:
            raise RuntimeError("step() called on a finished game")

        rots = PIECES[self.current]
        illegal = not (0 <= rotation < len(rots))
        rot = rots[rotation] if not illegal else None
        if not illegal and not (0 <= column < rot.n_placements):
            illegal = True

        if not illegal:
            col_top = self._col_top()
            top_row = self._drop_row(rot, column, col_top)
            if top_row < 0:
                illegal = True

        if illegal:
            # Safety path only: agents must choose from legal_placements().
            self.game_over = True
            return StepInfo(0, True, 0.0, 0, None)

        post, lines, landing_height, eroded = self._apply(rot, column, top_row)
        self.rows = post
        self.lines += lines
        self.pieces += 1

        board6 = board_features_batch(np.array([post], dtype=np.uint16))[0]
        features = (float(landing_height), float(eroded)) + tuple(
            int(x) for x in board6
        )

        # Draw next piece; game over if it has no legal placement.
        self.current = self.queue.popleft()
        self._fill_queue()
        if not self.legal_placements():
            self.game_over = True

        return StepInfo(lines, self.game_over, landing_height, eroded, features)

    def clone(self) -> "TetrisEngine":
        e = TetrisEngine.__new__(TetrisEngine)
        e.rng = self.rng.clone()
        e.bag = SevenBag(e.rng)
        e.bag._bag = list(self.bag._bag)
        e.preview = self.preview
        e.rows = list(self.rows)
        e.queue = deque(self.queue)
        e.current = self.current
        e.lines = self.lines
        e.pieces = self.pieces
        e.game_over = self.game_over
        return e
