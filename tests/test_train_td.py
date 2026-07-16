"""Validate the runs/ contract a smoke TD run writes (PLAN.md §6, §8).

Runs `train_td.py --smoke` once in-process, then asserts the schema of every
artifact: config.json (incl. milestone mapping), every metrics.jsonl line,
replays round-trip determinism, torch checkpoints incl. nn_step_0 and the four
milestones, and that the warm-start regression loss decreased.
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
from tetris.runio import METRICS_FIELDS

import train_td  # noqa: E402


@pytest.fixture(scope="module")
def smoke_run() -> Path:
    assert train_td.main(["--smoke"]) == 0
    run_dir = _ROOT / "runs" / "td_smoke"
    assert run_dir.is_dir()
    return run_dir


def test_config_has_milestones_and_td_phase(smoke_run: Path):
    cfg = json.loads((smoke_run / "config.json").read_text())
    assert cfg["phase"] == "td"
    assert cfg["config"]["algorithm"] == "td0-afterstate"
    milestones = cfg["milestones"]
    assert set(milestones) == {"0", "10", "30", "60", "100"}
    assert milestones["0"]["checkpoint"] == "nn_step_0"
    for m in milestones.values():
        assert (smoke_run / "checkpoints" / f"{m['checkpoint']}.pt").is_file()


def test_metrics_schema_and_finite(smoke_run: Path):
    lines = (smoke_run / "metrics.jsonl").read_text().splitlines()
    assert lines
    seen_loss = False
    for raw in lines:
        row = json.loads(raw)
        assert set(row.keys()) == set(METRICS_FIELDS)
        assert row["phase"] == "td"
        assert row["pieces_trained"] is not None
        for f in ("eval_median_lines", "eval_mean_lines", "epsilon", "beta"):
            assert row[f] is not None
        if row["loss"] is not None:
            import math

            assert math.isfinite(row["loss"])
            seen_loss = True
    assert seen_loss, "no TD loss ever logged"


def test_learning_curve_rises_off_the_floor(smoke_run: Path):
    rows = [json.loads(l) for l in (smoke_run / "metrics.jsonl").read_text().splitlines()]
    first = rows[0]["eval_median_lines"]  # untrained (nn_step_0)
    best = max(r["eval_median_lines"] for r in rows)
    assert best > first, "eval median never exceeded the untrained floor"


def test_warmstart_loss_decreased(smoke_run: Path):
    ws = json.loads((smoke_run / "checkpoints" / "warmstart.json").read_text())
    losses = ws["epoch_losses"]
    assert len(losses) == 3
    assert losses[-1] < losses[0]


def test_replays_round_trip(smoke_run: Path):
    index = json.loads((smoke_run / "replays" / "index.json").read_text())
    assert index
    for entry in index:
        replay = json.loads((smoke_run / "replays" / entry["file"]).read_text())
        lines, pieces = replay_moves(replay["seed"], replay["moves"])
        assert lines == replay["final"]["lines"]
        assert pieces == replay["final"]["pieces"]


def test_checkpoints_loadable(smoke_run: Path):
    import torch

    from tetris.model import ValueNet

    for name in ("nn_step_0", "nn_step_6000"):
        state = torch.load(
            smoke_run / "checkpoints" / f"{name}.pt", weights_only=False
        )
        model = ValueNet()
        model.load_state_dict(state["model_state"])  # arch matches
