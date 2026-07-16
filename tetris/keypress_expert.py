"""Keypress-level expert / teacher (PLAN2.md §4).

Given a :class:`~tetris.frame_env.FrameEnv` at a piece's spawn, the expert:

1. Enumerates the v1 straight-drop placements for the active piece on the frame
   layer's board (reusing the frozen v1 engine's ``candidate_features`` — the
   frame board is a plain 20-row bitboard, identical representation).
2. Filters to REACHABLE placements. For each candidate ``(rot, col)`` it builds
   the *naive* keypress script — noops until the piece is FULLY VISIBLE (every
   cell at row >= 0; camera-faithfulness, §4 amendment — the wait lives in the
   script driver, see :func:`simulate_script` / :class:`ExpertPlayer`), then all
   rotations, then all slides, then noops until lock — and forward-simulates it
   on a CLONE of the frame env
   (exact frame-layer physics: 3 ticks per decision, 1 gravity row per 24
   ticks). A placement is reachable iff the sim locks at exactly ``(rot, col)``
   with ``tuck=False`` (i.e. the piece comes to rest at its v1 straight-drop
   pose). The clone carries the true tick/gravity phase, so reachability
   accounts for the piece descending while the script is being pressed.
3. Scores reachable afterstates with the td_v1 ValueNet
   (``q = r + gamma * V(after)``, eval semantics ``beta=0``, matching
   :class:`~tetris.agents.ValueNetAgent`) or, with ``--teacher cem``, the linear
   CEM weights (``w . features``). One batched forward scores every candidate.
4. Emits the naive script for the best reachable placement.

To keep planning cheap, candidates are scored once (batched) and reachability is
checked lazily from the highest-scored placement down; the first reachable one
wins. This means the common case runs a single forward-sim per decision.

:class:`ExpertPlayer` wraps this as a per-decision-tick policy for Phase C/D
dataset generation: ``reset(env)`` then ``act(env) -> action`` at every decision
tick; it replans once per spawn and streams the script one action per tick.

Determinism: enumeration order, argmax tie-breaks (first in enumeration order),
and the frame layer are all deterministic, so a given seed always yields the
same game.
"""

from __future__ import annotations

import argparse
import statistics
from collections import deque
from pathlib import Path

import numpy as np

from tetris.agents import decision_rewards
from tetris.engine import PIECES, TetrisEngine
from tetris.evaluation import p10
from tetris.features import WIDTH
from tetris.frame_env import (
    LEFT,
    NOOP,
    RIGHT,
    ROT_CCW,
    ROT_CW,
    FrameEnv,
)
from tetris.rng import SevenBag

_REPO = Path(__file__).resolve().parent.parent
TD_V1_CHECKPOINT = _REPO / "runs" / "td_v1" / "checkpoints" / "nn_step_2000000.pt"
CEM_V1_WEIGHTS = _REPO / "runs" / "cem_v1" / "checkpoints" / "cem_final.json"


# --------------------------------------------------------------------------
# Frame-env cloning (local, so frame_env.py stays untouched by Phase B).
# --------------------------------------------------------------------------


def clone_env(env: FrameEnv) -> FrameEnv:
    """Deep-copy a :class:`FrameEnv` (piece + board + rng + tick/gravity phase).

    Mirrors :meth:`tetris.engine.TetrisEngine.clone`: the RNG is cloned and a
    fresh :class:`SevenBag` is wired to it with a copied bag, so the clone draws
    the identical future piece sequence while remaining fully independent.
    """
    e = FrameEnv.__new__(FrameEnv)
    e.rng = env.rng.clone()
    e.bag = SevenBag(e.rng)
    e.bag._bag = list(env.bag._bag)
    e.preview = env.preview
    e.rows = list(env.rows)
    e.queue = deque(env.queue)
    e.piece = env.piece
    e.rot = env.rot
    e.col = env.col
    e.row = env.row
    e.lines = env.lines
    e.pieces = env.pieces
    e.score = env.score
    e.tick_count = env.tick_count
    e.gravity_counter = env.gravity_counter
    e.game_over = env.game_over
    e._pending = env._pending
    return e


# --------------------------------------------------------------------------
# Naive keypress script + reachability
# --------------------------------------------------------------------------


def fully_visible(env: FrameEnv) -> bool:
    """True iff every active-piece cell is on the visible board (row >= 0).

    Camera-faithfulness (PLAN2.md §4 amendment): the expert may not act until
    this holds — before that the camera has not rendered the piece at all (§1:
    cells above the board are not drawn) and the preview already shows the NEXT
    piece, so any earlier action would be conditioned on invisible state.
    Bounding boxes are tight (the top bbox row always contains a cell), so the
    exact per-cell test reduces to ``env.row >= 0``; we keep the honest form.
    """
    return all(env.row + ro >= 0 for ro, _ in PIECES[env.piece][env.rot].cells)


def naive_script(piece: int, target_rot: int, target_col: int) -> list[int]:
    """The naive per-decision-tick action list for ``(target_rot, target_col)``:
    all rotations first, then all slides. Noops-until-lock are implicit (the
    driver emits ``NOOP`` once the list is exhausted).

    Rotation direction is the one needing fewer presses (cw on a tie). The
    post-rotation column is derived by replaying the frame layer's rotation
    clamp (``col`` clamped to ``[0, WIDTH - width]`` at each step), from which
    the slide count/direction follows.
    """
    n = len(PIECES[piece])
    cw = target_rot % n
    ccw = (n - target_rot) % n
    if cw <= ccw:
        rot_action, count = ROT_CW, cw
    else:
        rot_action, count = ROT_CCW, ccw
    script = [rot_action] * count

    # Replay the rotation clamp to find the column after the rotations.
    rot0 = PIECES[piece][0]
    col = (WIDTH - rot0.width) // 2
    rot = 0
    for _ in range(count):
        rot = (rot + 1) % n if rot_action == ROT_CW else (rot - 1) % n
        w = PIECES[piece][rot].width
        col = min(max(col, 0), WIDTH - w)

    delta = target_col - col
    slide_action = RIGHT if delta > 0 else LEFT
    script += [slide_action] * abs(delta)
    return script


def current_pose_script(piece: int, cur_rot: int, cur_col: int,
                        target_rot: int, target_col: int) -> list[int]:
    """Generalization of :func:`naive_script` to an ARBITRARY current pose:
    the per-decision-tick action list that takes the piece from ``(cur_rot,
    cur_col)`` to ``(target_rot, target_col)`` — all rotations first (shorter
    direction, cw on a tie), replaying the frame layer's column clamp, then all
    slides. Noops-until-lock are implicit (the driver emits ``NOOP`` once the
    list is exhausted). At spawn (``cur_rot=0``, ``cur_col=spawn_col``) this
    equals :func:`naive_script`.

    This is the DAgger primitive: for a student-visited mid-flight state, the
    expert re-plans from the CURRENT pose, so the correct label is the first
    action of *this* script (a spawn-time script would mislabel it).
    """
    n = len(PIECES[piece])
    cw = (target_rot - cur_rot) % n
    ccw = (cur_rot - target_rot) % n
    if cw <= ccw:
        rot_action, count = ROT_CW, cw
    else:
        rot_action, count = ROT_CCW, ccw
    script = [rot_action] * count

    # Replay the rotation clamp from the current column to find the post-rotation
    # column (mirrors FrameEnv._do_action's clamp; collisions are resolved by the
    # forward-sim in the caller, not here).
    col = cur_col
    rot = cur_rot
    for _ in range(count):
        rot = (rot + 1) % n if rot_action == ROT_CW else (rot - 1) % n
        w = PIECES[piece][rot].width
        col = min(max(col, 0), WIDTH - w)

    delta = target_col - col
    slide_action = RIGHT if delta > 0 else LEFT
    script += [slide_action] * abs(delta)
    return script


def relabel_action(env: FrameEnv, teacher) -> int:
    """DAgger expert label for the CURRENT (possibly mid-flight) pose of ``env``.

    Enumerates the current board's placements, scores them with ``teacher``, and
    walks them high-score first; for each it builds the :func:`current_pose_script`
    (rotations, slides, then waits) and forward-simulates it on a CLONE from the
    current tick/gravity phase. The first placement that the current pose can
    still reach at its exact ``(rot, col)`` straight-drop pose (``tuck=False``)
    wins, and its script's FIRST action is returned as the label (``NOOP`` when
    the script is empty — already positioned — or nothing is reachable).

    Camera-faithfulness (§4 amendment): while the piece is NOT fully visible the
    label is ``NOOP`` unconditionally — the expert never acts on state the
    camera cannot see, and neither may the labels the student imitates.

    ``env`` is never mutated. This is the correct DAgger relabel: it reflects
    what the expert would press RIGHT NOW given where the student left the piece,
    which a spawn-time script cannot express once the piece has moved.
    """
    return _DaggerRelabelCore(teacher).relabel(env)


class _DaggerRelabelCore:
    """Relabeler with an optional per-piece teacher-score cache.

    The teacher score depends only on the board + active piece id, both constant
    across a single piece's ~150 decision frames, so scoring once per piece and
    reusing it across relabels is a ~150x saving on the neural forward. Only the
    cheap current-pose reachability forward-sim runs per frame. Reset per game."""

    __slots__ = ("teacher", "_pieces", "_placements", "_order")

    def __init__(self, teacher):
        self.teacher = teacher
        self.reset()

    def reset(self) -> None:
        self._pieces = None
        self._placements: list | None = None
        self._order = None

    def relabel(self, env: FrameEnv) -> int:
        if not fully_visible(env):  # §4 amendment: no label before the camera sees it
            return NOOP
        if self._placements is None or env.pieces != self._pieces:
            scored = self.teacher.scores(_placement_engine(env))
            self._pieces = env.pieces
            if scored is None:
                self._placements, self._order = [], []
            else:
                self._placements, scores = scored
                self._order = np.argsort(-scores, kind="stable")
        for idx in self._order:
            rot, col = self._placements[int(idx)]
            script = current_pose_script(env.piece, env.rot, env.col, rot, col)
            lock = simulate_script(clone_env(env), script)
            if (lock is not None and lock["r"] == rot and lock["c"] == col
                    and not lock["tuck"]):
                return script[0] if script else NOOP
        return NOOP


class DaggerRelabeler(_DaggerRelabelCore):
    """Public per-game relabeler (see :class:`_DaggerRelabelCore`)."""


def simulate_script(env: FrameEnv, script: list[int],
                    wait_visible: bool = True) -> dict | None:
    """Forward-simulate ``script`` on ``env`` (mutated; pass a clone) exactly as
    the real driver would: ``NOOP`` at every decision tick until the piece is
    fully visible (camera-faithfulness, §4 amendment — the spec default), then
    one script action per decision tick, ``NOOP`` once the script is exhausted,
    ticking to the lock. ``wait_visible=False`` gives the raw pre-amendment
    driver (kept for pose-relative sims that start from a visible pose).

    Returns the frame layer's lock-event dict (``r`` = rotation, ``c`` = column,
    ``tuck`` flag) for the piece active at call time, or ``None`` if the game
    ends first.
    """
    i = 0
    while not env.game_over:
        if env.is_decision_tick:
            if wait_visible and not fully_visible(env):
                env.apply_action(NOOP)
            else:
                env.apply_action(script[i] if i < len(script) else NOOP)
                i += 1
        lock = env.tick()
        if lock is not None:
            return lock
    return None


def is_reachable(env: FrameEnv, target_rot: int, target_col: int) -> bool:
    """True iff the naive script for ``(target_rot, target_col)`` lands the
    active piece at that exact v1 straight-drop pose (``tuck=False``) under real
    frame-layer gravity. Simulated on a clone; ``env`` is not mutated."""
    script = naive_script(env.piece, target_rot, target_col)
    lock = simulate_script(clone_env(env), script)
    if lock is None:
        return False
    return lock["r"] == target_rot and lock["c"] == target_col and not lock["tuck"]


# --------------------------------------------------------------------------
# Teachers
# --------------------------------------------------------------------------


def load_cem_weights(path: str | Path = CEM_V1_WEIGHTS) -> np.ndarray:
    import json

    with open(path) as f:
        data = json.load(f)
    weights = data["weights"] if isinstance(data, dict) else data
    return np.asarray(weights, dtype=np.float64)


class _TDTeacher:
    """Scores afterstates with the td_v1 ValueNet: ``q = r + gamma * V``."""

    def __init__(self, checkpoint, device: str = "cpu", gamma: float = 0.95):
        from tetris.export import load_valuenet

        self.model = load_valuenet(checkpoint, device=device)
        self.device = device
        self.gamma = float(gamma)

    def scores(self, engine) -> tuple[list, np.ndarray] | None:
        import torch

        from tetris.model import boards_to_tensor

        res = decision_rewards(engine, 0.0)  # beta=0 eval semantics
        if res is None:
            return None
        placements, afters, r = res
        with torch.no_grad():
            v = self.model(boards_to_tensor(afters, self.device))
            v = v.detach().cpu().numpy().reshape(-1)
        return placements, r + self.gamma * v


class _CEMTeacher:
    """Scores placements with the linear CEM weights: ``w . features``."""

    def __init__(self, weights_path=CEM_V1_WEIGHTS):
        self.weights = load_cem_weights(weights_path)

    def scores(self, engine) -> tuple[list, np.ndarray] | None:
        placements, feats, _ = engine.candidate_features()
        if not placements:
            return None
        return placements, feats @ self.weights


def make_teacher(kind: str, checkpoint=None, device: str = "cpu"):
    if kind == "td":
        return _TDTeacher(checkpoint or TD_V1_CHECKPOINT, device=device)
    if kind == "cem":
        return _CEMTeacher(checkpoint or CEM_V1_WEIGHTS)
    raise ValueError(f"unknown teacher: {kind}")


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


class PlanResult:
    """A chosen placement + its naive script + planning diagnostics."""

    __slots__ = (
        "script",
        "target",
        "reachable",
        "n_candidates",
        "top_unreachable",
        "reachable_all",
    )

    def __init__(self, script, target, reachable, n_candidates,
                 top_unreachable, reachable_all):
        self.script = script  # list[int] per-decision-tick actions (noop-padded)
        self.target = target  # (rot, col) or None
        self.reachable = reachable  # True iff `target` was found reachable (not a fallback)
        self.n_candidates = n_candidates
        self.top_unreachable = top_unreachable  # top-scored placement was unreachable
        self.reachable_all = reachable_all  # (#reachable, #candidates) if full-checked else None


def _placement_engine(env: FrameEnv) -> TetrisEngine:
    """A throwaway v1 engine sharing the frame env's board + active piece, used
    only for placement enumeration / feature+afterstate computation."""
    eng = TetrisEngine(seed=0)
    eng.rows = list(env.rows)
    eng.current = env.piece
    return eng


def plan(env: FrameEnv, teacher, check_all_reachable: bool = False) -> PlanResult:
    """Choose the best reachable placement for the active piece and return its
    naive script.

    Candidates are scored once (batched inside ``teacher``) and reachability is
    checked lazily from the highest score down; the first reachable placement
    wins. ``check_all_reachable`` additionally checks every candidate to report
    the overall reachable fraction (diagnostic only; costs one forward-sim per
    candidate).
    """
    eng = _placement_engine(env)
    scored = teacher.scores(eng)
    if scored is None:
        return PlanResult([], None, False, 0, False,
                          (0, 0) if check_all_reachable else None)
    placements, scores = scored
    order = np.argsort(-scores, kind="stable")  # high score first, stable ties

    reachable_all = None
    if check_all_reachable:
        n_reach = sum(1 for (rot, col) in placements if is_reachable(env, rot, col))
        reachable_all = (n_reach, len(placements))

    chosen = None
    top_unreachable = False
    for rank, idx in enumerate(order):
        rot, col = placements[idx]
        if is_reachable(env, rot, col):
            chosen = (rot, col)
            break
        if rank == 0:
            top_unreachable = True

    if chosen is None:
        # No reachable placement (near top-out). Best effort: naive script for
        # the top-scored placement; the game is almost certainly ending.
        rot, col = placements[int(order[0])]
        return PlanResult(
            naive_script(env.piece, rot, col),
            (rot, col),
            False,
            len(placements),
            top_unreachable,
            reachable_all,
        )

    return PlanResult(
        naive_script(env.piece, *chosen),
        chosen,
        True,
        len(placements),
        top_unreachable,
        reachable_all,
    )


# --------------------------------------------------------------------------
# Per-decision-tick player (Phase C/D interface)
# --------------------------------------------------------------------------


class ExpertPlayer:
    """Stateful per-piece keypress policy over a :class:`FrameEnv`.

    Call :meth:`reset` once at game start, then :meth:`act` on every decision
    tick; it replans at each spawn and streams the naive script one action per
    tick (``NOOP`` after the script is exhausted). ``check_all_reachable`` turns
    on the full per-candidate reachability diagnostic (used by the eval harness).
    """

    def __init__(self, teacher, check_all_reachable: bool = False):
        self.teacher = teacher
        self.check_all_reachable = check_all_reachable
        self.reset(None)

    def reset(self, env: FrameEnv | None) -> "ExpertPlayer":
        self._plan_pieces = -1  # forces a replan on first act
        self._script: list[int] = []
        self._i = 0
        self.last_plan: PlanResult | None = None
        return self

    def act(self, env: FrameEnv) -> int:
        if env.pieces != self._plan_pieces:
            self.last_plan = plan(env, self.teacher, self.check_all_reachable)
            self._script = self.last_plan.script
            self._i = 0
            self._plan_pieces = env.pieces
        # Camera-faithfulness (§4 amendment): emit NOOP — without consuming the
        # script — until the piece is fully visible. Planning above happens at
        # spawn (the board/next-preview it reads are unchanged by the fall), but
        # no rotation/slide is pressed before the camera has seen the piece.
        if not fully_visible(env):
            return NOOP
        a = self._script[self._i] if self._i < len(self._script) else NOOP
        self._i += 1
        return a


# --------------------------------------------------------------------------
# Headless real-time eval
# --------------------------------------------------------------------------


class EvalStats:
    __slots__ = ("lines", "pieces", "top_unreachable", "decisions",
                 "reach_ok", "reach_total")

    def __init__(self):
        self.lines: list[int] = []
        self.pieces: list[int] = []
        self.top_unreachable = 0
        self.decisions = 0
        self.reach_ok = 0
        self.reach_total = 0


def play_game(player: ExpertPlayer, seed: int, max_pieces: int,
              stats: EvalStats, sample_reach: int = 0) -> tuple[int, int]:
    """Play one headless real-time game to game-over or the piece cap. Ticks are
    simulated (no wall-clock sleep). ``sample_reach`` (>0) turns on the full
    reachability diagnostic every Nth decision."""
    env = FrameEnv(seed=seed)
    player.reset(env)
    dec = 0
    while not env.game_over and env.pieces < max_pieces:
        if env.is_decision_tick:
            new_piece = env.pieces != player._plan_pieces
            player.check_all_reachable = bool(sample_reach) and (dec % sample_reach == 0)
            action = player.act(env)
            if new_piece and player.last_plan is not None:
                p = player.last_plan
                stats.decisions += 1
                if p.top_unreachable:
                    stats.top_unreachable += 1
                if p.reachable_all is not None:
                    stats.reach_ok += p.reachable_all[0]
                    stats.reach_total += p.reachable_all[1]
            env.apply_action(action)
            dec += 1
        env.tick()
    return env.lines, env.pieces


def evaluate(games: int, seed: int, max_pieces: int, teacher_kind: str = "td",
             checkpoint=None, device: str = "cpu", sample_reach: int = 20) -> EvalStats:
    teacher = make_teacher(teacher_kind, checkpoint, device)
    player = ExpertPlayer(teacher)
    stats = EvalStats()
    for i in range(games):
        lines, pieces = play_game(player, seed + i, max_pieces, stats, sample_reach)
        stats.lines.append(lines)
        stats.pieces.append(pieces)
    return stats


def _report(stats: EvalStats, games: int, seed: int, max_pieces: int,
            teacher_kind: str, elapsed: float) -> None:
    for i, (ln, pc) in enumerate(zip(stats.lines, stats.pieces)):
        print(f"game {i:>3}  seed={seed + i:>7}  lines={ln:>7}  pieces={pc:>7}")
    print("-" * 56)
    print(f"teacher                : {teacher_kind}")
    print(f"games                  : {games}   max_pieces={max_pieces}")
    print(f"lines  median          : {statistics.median(stats.lines):.1f}")
    print(f"lines  mean            : {statistics.mean(stats.lines):.1f}")
    print(f"lines  p10             : {p10(stats.lines):.1f}")
    print(f"pieces median          : {statistics.median(stats.pieces):.1f}")
    frac_top = (stats.top_unreachable / stats.decisions) if stats.decisions else 0.0
    print(f"chosen-was-top-unreach : {frac_top:.4f}  "
          f"({stats.top_unreachable}/{stats.decisions} decisions)")
    if stats.reach_total:
        print(f"overall reachable frac : {stats.reach_ok / stats.reach_total:.4f}  "
              f"({stats.reach_ok}/{stats.reach_total} placements, sampled)")
    print(f"wall-clock             : {elapsed:.1f}s")


def main(argv=None) -> int:
    import time

    ap = argparse.ArgumentParser(description="Keypress expert real-time eval (PLAN2.md §4)")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=900000, help="base seed; game i uses seed+i")
    ap.add_argument("--max-pieces", type=int, default=10000)
    ap.add_argument("--teacher", choices=["td", "cem"], default="td")
    ap.add_argument("--checkpoint", default=None, help="override teacher checkpoint path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample-reach", type=int, default=20,
                    help="full reachability diagnostic every Nth decision (0=off)")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    stats = evaluate(args.games, args.seed, args.max_pieces, args.teacher,
                     args.checkpoint, args.device, args.sample_reach)
    _report(stats, args.games, args.seed, args.max_pieces, args.teacher,
            time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
