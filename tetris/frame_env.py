"""Frame layer with physical-pose locking (PLAN2.md §1, amended).

Real-time keypress-level Tetris on the v1 board representation. The frame layer
owns its board (20 rows of 10-bit ints, row 0 top) and applies locks itself at
the piece's TRUE PHYSICAL POSE — what the camera sees is what locks. It reuses
the frozen v1 building blocks read-only: piece rotation tables
(:data:`tetris.engine.PIECES`), :class:`~tetris.rng.Mulberry32` +
:class:`~tetris.rng.SevenBag` piece supply (identical queue discipline to the
v1 engine, so a given seed yields the v1 piece sequence), and the v1
``CLEAR_POINTS`` scoring table.

Semantics (frozen §1):

* Logic runs at 30 Hz (ticks).
* Gravity: the active piece descends 1 row every ``GRAVITY_PERIOD`` (24) ticks.
* Decisions: an action may be emitted every ``DECISION_PERIOD`` (3rd) tick
  (10 Hz "hand rate"); intermediate ticks advance gravity only.
* Actions: ``noop``, ``left``, ``right``, ``rot_cw``, ``rot_ccw`` — a slide
  moves 1 column if the destination cells are collision-free, else a silent
  no-op; rotation keeps the current top-left anchor with the column clamped to
  ``[0, 10 - width]`` and fails (silent no-op) on collision. No kicks.
* Spawn: rotation 0, column ``floor((10 - width) / 2)``, bounding-box bottom at
  board row -1 (fully above the board), on the tick after lock. A spawn pose
  that collides ends the game.
* Lock: when a gravity descent would collide, the piece locks at its current
  physical pose: place cells, remove full rows shifting above down, score via
  ``CLEAR_POINTS``, draw the next piece from the same 7-bag. Game over iff any
  locked cell has row < 0 (the board is left unchanged, mirroring the v1
  engine's illegal-placement outcome bit-for-bit) or the next spawn pose
  collides.

Tucks and the v1-consistency invariant
--------------------------------------
Because a slide only checks the *destination* cells and there are 8 decision
ticks per gravity row, a falling piece can slide sideways under an overhang and
rest deeper than the v1 straight drop of its (rot, col) — a "tuck". Locks
resolve at that physical pose (no board teleport); the lock event carries a
``tuck`` flag (``True`` iff the lock row differs from the straight-drop row).

The v1-consistency invariant — fixture-tested here and in
``tests_js/parity_v2.test.mjs`` — is: whenever the lock pose equals the
straight-drop pose (every non-tuck lock, the overwhelmingly common case), the
board/lines transition is bit-identical to v1 ``engine.step(r, c)``. Straight
drop is the highest reachable rest, so ``straight_drop_row <= lock_row`` always
holds (asserted at every lock).
"""

from __future__ import annotations

from collections import deque

from tetris.engine import CLEAR_POINTS, PIECES
from tetris.features import FULL_ROW, HEIGHT, WIDTH
from tetris.rng import Mulberry32, SevenBag

# Frozen timing constants (PLAN2.md §1).
TICK_HZ = 30
GRAVITY_PERIOD = 24  # ticks per 1-row gravity descent
DECISION_PERIOD = 3  # a decision may be emitted every 3rd tick

# Action space (PLAN2.md §1). Index order is frozen.
ACTIONS = ("noop", "left", "right", "rot_cw", "rot_ccw")
NOOP, LEFT, RIGHT, ROT_CW, ROT_CCW = range(5)


class FrameEnv:
    """Real-time frame layer with physical-pose locking (module docstring)."""

    def __init__(self, seed: int = 0, preview: int = 5):
        self.reset(seed, preview)

    # -- lifecycle -----------------------------------------------------------

    def reset(self, seed: int, preview: int = 5) -> "FrameEnv":
        self.rng = Mulberry32(seed)
        self.bag = SevenBag(self.rng)
        self.preview = preview
        self.rows: list[int] = [0] * HEIGHT
        # Piece supply: identical queue discipline to the v1 engine, so the
        # piece sequence for a seed matches v1 exactly.
        self.queue: deque[int] = deque()
        self._fill_queue()
        self.piece: int = self.queue.popleft()
        self._fill_queue()
        self.lines = 0
        self.pieces = 0
        self.score = 0  # sum of CLEAR_POINTS[lines] * 100, v1 demo convention
        self.tick_count = 0
        self.gravity_counter = 0
        self.game_over = False
        self._pending = NOOP
        self._spawn()
        # Spawn pose is fully above an empty board, so this is defensive only.
        if self._collides(self.piece, self.rot, self.col, self.row):
            self.game_over = True
        return self

    def _fill_queue(self) -> None:
        while len(self.queue) < self.preview:
            self.queue.append(self.bag.next_piece())

    def preview_pieces(self) -> list[int]:
        """The next `preview` piece indices (does not include the active piece)."""
        return list(self.queue)[: self.preview]

    def _spawn(self) -> None:
        rot0 = PIECES[self.piece][0]
        self.rot = 0
        self.col = (WIDTH - rot0.width) // 2
        self.row = -rot0.height  # bounding-box bottom at board row -1

    # -- introspection -------------------------------------------------------

    @property
    def is_decision_tick(self) -> bool:
        return self.tick_count % DECISION_PERIOD == 0

    def pose(self) -> tuple[int, int, int, int]:
        """Active piece as (piece_id, rot, col, row)."""
        return (self.piece, self.rot, self.col, self.row)

    # -- geometry ------------------------------------------------------------

    def _collides(self, piece: int, rot: int, col: int, row: int) -> bool:
        """True if placing `piece` at (rot, col, row) overlaps a wall, the
        floor, or a filled stack cell. Cells above the board (row < 0) are free."""
        rows = self.rows
        for ro, co in PIECES[piece][rot].cells:
            rr = row + ro
            cc = col + co
            if cc < 0 or cc >= WIDTH:
                return True
            if rr >= HEIGHT:
                return True
            if rr >= 0 and (rows[rr] >> cc) & 1:
                return True
        return False

    def _straight_drop_row(self, rot_idx: int, col: int) -> int:
        """v1 straight-drop resting top-row for the active piece at (rot, col)
        on the current board (may be negative => v1-illegal)."""
        rot = PIECES[self.piece][rot_idx]
        t = HEIGHT
        for pc in range(rot.width):
            c = col + pc
            top = HEIGHT
            for r in range(HEIGHT):
                if (self.rows[r] >> c) & 1:
                    top = r
                    break
            v = top - 1 - rot.bottom[pc]
            if v < t:
                t = v
        return t

    # -- actions -------------------------------------------------------------

    def apply_action(self, action: int) -> None:
        """Queue an action for the current decision tick. Only valid on a
        decision tick; the queued action is consumed by the next ``tick()``."""
        if not self.is_decision_tick:
            raise ValueError("apply_action is only valid on a decision tick")
        self._pending = action

    def _do_action(self, action: int) -> None:
        if action == NOOP:
            return
        if action == LEFT or action == RIGHT:
            nc = self.col + (-1 if action == LEFT else 1)
            if not self._collides(self.piece, self.rot, nc, self.row):
                self.col = nc
            return
        # Rotation: keep the top-left anchor, clamp column to [0, 10 - width].
        n = len(PIECES[self.piece])
        nr = (self.rot + 1) % n if action == ROT_CW else (self.rot - 1) % n
        w = PIECES[self.piece][nr].width
        nc = min(max(self.col, 0), WIDTH - w)
        if not self._collides(self.piece, nr, nc, self.row):
            self.rot = nr
            self.col = nc

    # -- tick ----------------------------------------------------------------

    def tick(self) -> dict | None:
        """Advance one tick. Returns a lock-event dict on the tick a piece
        locks, else ``None``."""
        if self.game_over:
            self.tick_count += 1
            return None

        if self.tick_count % DECISION_PERIOD == 0:
            self._do_action(self._pending)
        self._pending = NOOP

        lock = None
        self.gravity_counter += 1
        if self.gravity_counter >= GRAVITY_PERIOD:
            self.gravity_counter = 0
            if self._collides(self.piece, self.rot, self.col, self.row + 1):
                lock = self._lock_and_spawn()
            else:
                self.row += 1

        self.tick_count += 1
        return lock

    def _lock_and_spawn(self) -> dict:
        rot_idx = self.rot
        col = self.col
        row = self.row
        rot = PIECES[self.piece][rot_idx]

        # Straight drop is the highest reachable rest; deeper == tuck.
        drop = self._straight_drop_row(rot_idx, col)
        assert drop <= row, (
            f"straight-drop invariant violated: drop row {drop} > lock row "
            f"{row} (piece {self.piece} rot {rot_idx} col {col})"
        )
        tuck = drop != row

        # Any locked cell above the board => game over. The bbox top row always
        # contains a cell (tight bounding boxes), so row < 0 is the exact test.
        # Board left unchanged — bit-identical to the v1 illegal-step outcome.
        if row < 0:
            self.game_over = True
            return {
                "tick": self.tick_count,
                "r": rot_idx,
                "c": col,
                "lines_after": self.lines,
                "tuck": tuck,
            }

        # Physical lock: place cells, clear full rows shifting above down
        # (identical row surgery to v1 engine._apply).
        for ro, co in rot.cells:
            self.rows[row + ro] |= 1 << (col + co)
        lines = 0
        for r in range(row, row + rot.height):
            if self.rows[r] == FULL_ROW:
                lines += 1
        if lines:
            kept = [v for v in self.rows if v != FULL_ROW]
            self.rows = [0] * lines + kept
        self.lines += lines
        self.score += CLEAR_POINTS[lines] * 100
        self.pieces += 1
        lock = {
            "tick": self.tick_count,
            "r": rot_idx,
            "c": col,
            "lines_after": self.lines,
            "tuck": tuck,
        }

        # Draw the next piece and spawn; a colliding spawn pose ends the game.
        self.piece = self.queue.popleft()
        self._fill_queue()
        self._spawn()
        if self._collides(self.piece, self.rot, self.col, self.row):
            self.game_over = True
        return lock
