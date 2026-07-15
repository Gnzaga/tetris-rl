"""Validate the runs/ contract a smoke CEM run writes (PLAN.md §6).

Runs `train_cem.py --smoke` once in-process, then asserts the schema of every
artifact: config.json keys, every metrics.jsonl line, replays/index.json ->
files, replay round-trip determinism, runs/index.json, checkpoints, tb/ events.
Also exercises RunWriter directly in a tmp dir for the low-level guarantees.
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from tetris.evaluation import replay_moves
from tetris.runio import METRICS_FIELDS, RunWriter

import train_cem  # noqa: E402


@pytest.fixture(scope="module")
def smoke_run() -> Path:
    """Produce runs/cem_smoke once for the whole module."""
    assert train_cem.main(["--smoke"]) == 0
    run_dir = _ROOT / "runs" / "cem_smoke"
    assert run_dir.is_dir()
    return run_dir


# -- smoke-run schema validation --------------------------------------------


def test_config_json_schema(smoke_run: Path):
    cfg = json.loads((smoke_run / "config.json").read_text())
    for key in ("run_name", "phase", "engine_version", "created_at",
                "python_version", "argv", "git_commit", "git_dirty", "config"):
        assert key in cfg, f"config.json missing {key}"
    assert cfg["run_name"] == "cem_smoke"
    assert cfg["phase"] == "cem"
    inner = cfg["config"]
    for key in ("smoke", "population", "elites", "generations", "eval_seeds"):
        assert key in inner
    assert inner["smoke"] is True


def test_metrics_jsonl_schema(smoke_run: Path):
    lines = (smoke_run / "metrics.jsonl").read_text().splitlines()
    assert lines, "metrics.jsonl is empty"
    for raw in lines:
        row = json.loads(raw)
        assert set(row.keys()) == set(METRICS_FIELDS), "metrics row schema drift"
        assert row["phase"] == "cem"
        assert row["pieces_trained"] is not None
        for field in ("eval_median_lines", "eval_mean_lines", "eval_p10_lines"):
            assert row[field] is not None


def test_replays_index_points_at_existing_files(smoke_run: Path):
    index = json.loads((smoke_run / "replays" / "index.json").read_text())
    assert index, "replays/index.json is empty"
    for entry in index:
        for key in ("file", "pieces_trained", "median_lines", "ts"):
            assert key in entry
        assert (smoke_run / "replays" / entry["file"]).is_file()


def test_replay_roundtrip_is_deterministic(smoke_run: Path):
    index = json.loads((smoke_run / "replays" / "index.json").read_text())
    for entry in index:
        replay = json.loads((smoke_run / "replays" / entry["file"]).read_text())
        assert replay["engine_version"] == "1"
        for key in ("seed", "moves", "final"):
            assert key in replay
        lines, pieces = replay_moves(replay["seed"], replay["moves"])
        assert lines == replay["final"]["lines"], "replay line count not reproducible"
        assert pieces == replay["final"]["pieces"], "replay piece count not reproducible"


def test_checkpoints_written(smoke_run: Path):
    ckpts = list((smoke_run / "checkpoints").glob("*.json"))
    names = {p.stem for p in ckpts}
    assert "cem_final" in names
    assert any(n.startswith("cem_gen_") for n in names)
    for p in ckpts:
        json.loads(p.read_text())  # each is valid JSON


def test_tensorboard_events_written(smoke_run: Path):
    events = list((smoke_run / "tb").glob("events.out.tfevents.*"))
    assert events, "no TensorBoard event files written"


def test_runs_index_has_entry(smoke_run: Path):
    entries = json.loads((_ROOT / "runs" / "index.json").read_text())
    entry = next((e for e in entries if e["name"] == "cem_smoke"), None)
    assert entry is not None
    for key in ("phase", "created", "updated", "pieces_trained",
                "best_median_lines", "num_replays"):
        assert key in entry
    assert entry["num_replays"] >= 1


# -- RunWriter unit guarantees (isolated tmp dir) ---------------------------


def test_runwriter_rejects_unknown_metric_field(tmp_path: Path):
    with RunWriter("t", {"x": 1}, phase="cem", root=tmp_path) as run:
        with pytest.raises(KeyError):
            run.log(pieces_trained=1, not_a_field=3)


def test_runwriter_overwrite_and_index(tmp_path: Path):
    with RunWriter("r", {"a": 1}, phase="cem", root=tmp_path) as run:
        run.log(pieces_trained=10, eval_median_lines=5.0, eval_mean_lines=5.0,
                eval_p10_lines=5.0)
        run.save_replay(seed=3, moves=[[0, 0]], final={"lines": 0, "pieces": 1},
                        pieces_trained=10, median_lines=5.0)
    # Overwrite clears prior artifacts.
    stale = tmp_path / "r" / "checkpoints" / "stale.json"
    stale.write_text("{}")
    with RunWriter("r", {"a": 2}, phase="cem", root=tmp_path) as run:
        run.log(pieces_trained=20, eval_median_lines=7.0, eval_mean_lines=7.0,
                eval_p10_lines=7.0)
    assert not stale.exists(), "overwrite did not clear stale run dir"
    entries = json.loads((tmp_path / "index.json").read_text())
    entry = next(e for e in entries if e["name"] == "r")
    assert entry["pieces_trained"] == 20
    assert entry["best_median_lines"] == 7.0
