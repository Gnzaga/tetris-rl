"""Frame layer over the v1 atomic engine (PLAN2.md §1).

The v1 :class:`~tetris.engine.TetrisEngine` remains ground truth for lock,
line-clear, and game-over. This layer adds real-time semantics on top of it:

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
  pose. The equivalent v1 placement is simply ``(rot, col)`` — under
  gravity-only descent (no soft drop, no reachable overhang tuck) the frame
  resting row is exactly the v1 straight-drop resting row. The frame layer
  calls ``engine.step(rot, col)`` so the board transition is v1-consistent by
  construction; an above-row-0 lock maps to an engine-illegal placement and
  ends the game.

Straight-drop invariant (and its one exception: tucks)
------------------------------------------------------
For the common case the frame resting row equals ``engine._drop_row(rot, col)``:
a piece descends straight down and rests exactly where a v1 straight drop would.

There is one reachable exception. Because a slide only checks the *destination*
cells for collisions (frozen §1 semantics) and there are 8 decision ticks per
gravity row, a piece falling down an open column can slide sideways *under* an
overhang into a covered cell — a "tuck". Straight-drop-only play does create
overhangs (S/Z/T pieces on uneven terrain), so tucks are genuinely reachable and
occur in ordinary random play. A tucked piece rests deeper than any v1 straight
drop of its final ``(rot, col)`` can reach, so the true, always-holding invariant
is ``engine._drop_row(rot, col) <= frame_row`` (straight drop is the *highest*
reachable rest); equality is the no-tuck case.

The frozen spec mandates locking via ``engine.step(rot, col)`` regardless, so the
board transition stays v1-consistent by construction in both cases — on a tuck
the engine locks the piece at the (higher) straight-drop row. The lock event
carries a ``tuck`` flag so this is observable. See ``tests/test_frame_env.py``
for the invariant test, a constructed tuck, and rotation/clamp/above-board edges.
"""

from __future__ import annotations

from tetris.engine import PIECES, TetrisEngine
from tetris.features import HEIGHT, WIDTH

# Frozen timing constants (PLAN2.md §1).
TICK_HZ = 30
GRAVITY_PERIOD = 24  # ticks per 1-row gravity descent
DECISION_PERIOD = 3  # a decision may be emitted every 3rd tick

# Action space (PLAN2.md §1). Index order is frozen.
ACTIONS = ("noop", "left", "right", "rot_cw", "rot_ccw")
NOOP, LEFT, RIGHT, ROT_CW, ROT_CCW = range(5)


class FrameEnv:
    """Real-time frame layer wrapping the v1 engine (see module docstring)."""

    def __init__(self, seed: int = 0, preview: int = 5):
        self.reset(seed, preview)

    # -- lifecycle -----------------------------------------------------------

    def reset(self, seed: int, preview: int = 5) -> "FrameEnv":
        self.engine = TetrisEngine(seed=seed, preview=preview)
        self.tick_count = 0
        self.gravity_counter = 0
        self.game_over = False
        self._pending = NOOP
        self.piece = self.engine.current
        self._spawn()
        # Spawn pose is fully above the board, so this is defensive only.
        if self._collides(self.piece, self.rot, self.col, self.row):
            self.game_over = True
        return self

    def _spawn(self) -> None:
        rot0 = PIECES[self.piece][0]
        self.rot = 0
        self.col = (WIDTH - rot0.width) // 2
        self.row = -rot0.height  # bounding-box bottom at board row -1

    # -- introspection -------------------------------------------------------

    @property
    def is_decision_tick(self) -> bool:
        return self.tick_count % DECISION_PERIOD == 0

    @property
    def rows(self) -> list[int]:
        return self.engine.rows

    @property
    def lines(self) -> int:
        return self.engine.lines

    @property
    def pieces(self) -> int:
        return self.engine.pieces

    def pose(self) -> tuple[int, int, int, int]:
        """Active piece as (piece_id, rot, col, row)."""
        return (self.piece, self.rot, self.col, self.row)

    # -- geometry ------------------------------------------------------------

    def _collides(self, piece: int, rot: int, col: int, row: int) -> bool:
        """True if placing `piece` at (rot, col, row) overlaps a wall, the
        floor, or a filled stack cell. Cells above the board (row < 0) are free."""
        rows = self.engine.rows
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
        rot = PIECES[self.piece][rot_idx]
        # Straight-drop invariant: the v1 straight-drop row is the highest
        # reachable rest, so it never sits below the frame resting row. Equality
        # is the no-tuck case; drop < row means the piece tucked under an
        # overhang (see module docstring). The lock always defers to
        # engine.step(rot, col) so the board transition stays v1-consistent.
        drop = self.engine._drop_row(rot, col, self.engine._col_top())
        assert drop <= self.row, (
            f"straight-drop invariant violated: engine drop row {drop} > frame "
            f"row {self.row} (piece {self.piece} rot {rot_idx} col {col})"
        )

        self.engine.step(rot_idx, col)
        lock = {
            "tick": self.tick_count,
            "r": rot_idx,
            "c": col,
            "lines_after": self.engine.lines,
            "tuck": drop != self.row,
        }
        if self.engine.game_over:
            self.game_over = True
            return lock

        self.piece = self.engine.current
        self._spawn()
        if self._collides(self.piece, self.rot, self.col, self.row):
            self.game_over = True
        return lock
