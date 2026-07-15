"""The ``runs/<run_name>/`` directory contract (PLAN.md §6).

Every trainer writes the same on-disk layout so the three observation surfaces
(the terminal rich table, ``tensorboard --logdir runs``, and the demo's Replay
tab) all work against a single format:

    runs/<run_name>/
      config.json          resolved config + git-ish metadata
      metrics.jsonl        one JSON object per eval/log event (schema below)
      tb/                  TensorBoard scalars mirroring metrics.jsonl
      checkpoints/         cem_gen_<g>.json / nn_step_<pieces>.pt
      replays/
        index.json         [{file, pieces_trained, median_lines, ts}, ...]
        replay_<pieces>.json   {engine_version, seed, moves, final:{lines,pieces}}

Additionally a repo-level ``runs/index.json`` lists every run with light
metadata so the demo's Replay tab can enumerate runs without probing
(PLAN.md §10). All writes go through :class:`RunWriter`, which is the single
authority for the contract — trainers never touch these paths directly.

Metrics schema (every metrics.jsonl line carries all keys; unused ones are
null): ``wall_time, phase, pieces_trained, loss, epsilon, beta,
eval_median_lines, eval_mean_lines, eval_p10_lines, eval_pieces_per_game,
pps_train``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

# Frozen engine/replay format version (mirrors scripts/gen_fixtures.py).
ENGINE_VERSION = "1"

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = _REPO_ROOT / "runs"

# Every metrics.jsonl object carries exactly these keys (null when unused), so
# downstream readers can rely on a stable schema.
METRICS_FIELDS: tuple[str, ...] = (
    "wall_time",
    "phase",
    "pieces_trained",
    "loss",
    "epsilon",
    "beta",
    "eval_median_lines",
    "eval_mean_lines",
    "eval_p10_lines",
    "eval_pieces_per_game",
    "pps_train",
)

# Metric field -> TensorBoard scalar tag. Only these numeric fields are mirrored
# (wall_time/phase/pieces_trained are metadata, not curves).
_TB_TAGS: dict[str, str] = {
    "loss": "train/loss",
    "epsilon": "train/epsilon",
    "beta": "train/beta",
    "pps_train": "train/pps",
    "eval_median_lines": "eval/median_lines",
    "eval_mean_lines": "eval/mean_lines",
    "eval_p10_lines": "eval/p10_lines",
    "eval_pieces_per_game": "eval/pieces_per_game",
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_metadata() -> dict[str, Any]:
    """Best-effort git commit/dirty state; never raises."""

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode != 0:
                return None
            return out.stdout.strip()
        except Exception:
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
    }


class RunWriter:
    """Owns one ``runs/<run_name>/`` directory and its contract.

    Use as a context manager so buffers flush and TensorBoard closes cleanly::

        with RunWriter("cem_smoke", config, phase="cem") as run:
            run.log(pieces_trained=1000, eval_median_lines=42.0, ...)
            run.save_json_checkpoint("cem_gen_0", {...})
            run.save_replay(seed=7, moves=[[0, 3], ...],
                            final={"lines": 42, "pieces": 150},
                            pieces_trained=1000, median_lines=42.0)
    """

    def __init__(
        self,
        run_name: str,
        config: dict[str, Any],
        phase: str,
        root: Path = RUNS_DIR,
        overwrite: bool = True,
    ):
        self.run_name = run_name
        self.phase = phase
        self.root = Path(root)
        self.run_dir = self.root / run_name
        self.start_time = time.time()

        if self.run_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"run dir already exists: {self.run_dir} (pass overwrite=True)"
                )
            shutil.rmtree(self.run_dir)

        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.replays_dir = self.run_dir / "replays"
        self.tb_dir = self.run_dir / "tb"
        for d in (self.checkpoints_dir, self.replays_dir, self.tb_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._write_config(config)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._metrics_f = open(self.metrics_path, "a", buffering=1)
        self._tb = SummaryWriter(str(self.tb_dir))
        self._tb_step = 0

        self.replay_index_path = self.replays_dir / "index.json"
        self._replay_index: list[dict[str, Any]] = []
        self._write_replay_index()

        self._best_median: float | None = None
        self._last_pieces_trained: int = 0
        self._upsert_runs_index(created=_utc_iso())

    # -- config --------------------------------------------------------------

    def _write_config(self, config: dict[str, Any]) -> None:
        payload = {
            "run_name": self.run_name,
            "phase": self.phase,
            "engine_version": ENGINE_VERSION,
            "created_at": _utc_iso(),
            "python_version": sys.version.split()[0],
            "argv": list(sys.argv),
            **_git_metadata(),
            "config": config,
        }
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    # -- metrics -------------------------------------------------------------

    def log(self, **fields: Any) -> None:
        """Append one metrics.jsonl row and mirror its scalars to TensorBoard.

        Unknown keys are rejected so the schema stays stable. Missing keys are
        written as null. `wall_time` defaults to seconds since run start and
        `phase` to this run's phase.
        """
        unknown = set(fields) - set(METRICS_FIELDS)
        if unknown:
            raise KeyError(f"unknown metrics fields: {sorted(unknown)}")

        row: dict[str, Any] = {k: None for k in METRICS_FIELDS}
        row.update(fields)
        if row["wall_time"] is None:
            row["wall_time"] = round(time.time() - self.start_time, 3)
        if row["phase"] is None:
            row["phase"] = self.phase

        self._metrics_f.write(json.dumps(row) + "\n")

        pt = row["pieces_trained"]
        step = int(pt) if pt is not None else self._tb_step
        self._tb_step = step + 1
        for field, tag in _TB_TAGS.items():
            val = row.get(field)
            if val is not None:
                self._tb.add_scalar(tag, float(val), step)

        if pt is not None:
            self._last_pieces_trained = int(pt)
        med = row.get("eval_median_lines")
        if med is not None and (self._best_median is None or med > self._best_median):
            self._best_median = float(med)
        self._upsert_runs_index()

    # -- checkpoints ---------------------------------------------------------

    def save_json_checkpoint(self, name: str, payload: dict[str, Any]) -> Path:
        """Write ``checkpoints/<name>.json`` (e.g. name='cem_gen_3')."""
        path = self.checkpoints_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        return path

    def save_torch_checkpoint(self, name: str, state: Any) -> Path:
        """Write ``checkpoints/<name>.pt`` via torch.save (e.g. 'nn_step_50000')."""
        import torch  # local import: JSON-only trainers need not load torch here

        path = self.checkpoints_dir / f"{name}.pt"
        torch.save(state, path)
        return path

    # -- replays -------------------------------------------------------------

    def save_replay(
        self,
        seed: int,
        moves: list[list[int]],
        final: dict[str, int],
        pieces_trained: int,
        median_lines: float,
    ) -> Path:
        """Write a move-list replay and register it in replays/index.json.

        `moves` is ``[[rotation, column], ...]``; `final` is
        ``{"lines": ..., "pieces": ...}``. The deterministic engine reconstructs
        the full game from ``seed`` + ``moves``.
        """
        filename = f"replay_{int(pieces_trained)}.json"
        payload = {
            "engine_version": ENGINE_VERSION,
            "seed": int(seed),
            "moves": [[int(r), int(c)] for r, c in moves],
            "final": {"lines": int(final["lines"]), "pieces": int(final["pieces"])},
        }
        path = self.replays_dir / filename
        with open(path, "w") as f:
            json.dump(payload, f)
            f.write("\n")

        self._replay_index.append(
            {
                "file": filename,
                "pieces_trained": int(pieces_trained),
                "median_lines": float(median_lines),
                "ts": _utc_iso(),
            }
        )
        self._write_replay_index()
        self._upsert_runs_index()
        return path

    def _write_replay_index(self) -> None:
        with open(self.replay_index_path, "w") as f:
            json.dump(self._replay_index, f, indent=2)
            f.write("\n")

    # -- runs/index.json -----------------------------------------------------

    def _upsert_runs_index(self, created: str | None = None) -> None:
        """Insert or update this run's entry in the repo-level runs/index.json."""
        index_path = self.root / "index.json"
        entries: list[dict[str, Any]] = []
        if index_path.exists():
            try:
                with open(index_path) as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, OSError):
                entries = []

        entry = next((e for e in entries if e.get("name") == self.run_name), None)
        if entry is None:
            entry = {"name": self.run_name, "created": created or _utc_iso()}
            entries.append(entry)
        elif created is not None:
            entry["created"] = created

        entry.update(
            {
                "phase": self.phase,
                "updated": _utc_iso(),
                "pieces_trained": self._last_pieces_trained,
                "best_median_lines": self._best_median,
                "num_replays": len(self._replay_index),
            }
        )

        with open(index_path, "w") as f:
            json.dump(entries, f, indent=2)
            f.write("\n")

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        try:
            self._metrics_f.flush()
            self._metrics_f.close()
        finally:
            self._tb.flush()
            self._tb.close()

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
