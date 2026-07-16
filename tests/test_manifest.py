"""demo/models/manifest.json schema check (PLAN.md §9).

Builds a manifest from the smoke runs into a tmp dir (skipping vendoring) and
validates its structure, types, and cross-references: milestone ONNX files exist,
linear weight arrays are length 8, the self-test has 3 boards of 20 rows with 3
expected values, and the training curve is non-empty with ascending
pieces_trained. Parity board count is reduced for test speed — the full 1,000-
board parity gate is exercised by ``scripts/export_demo.py`` itself.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.path.insert(0, str(_ROOT / "scripts"))
import export_demo  # noqa: E402

TD_RUN = "td_smoke"
CEM_RUN = "cem_smoke"

_EXPECTED_NN_IDS = ["nn_000", "nn_010", "nn_030", "nn_060", "nn_100"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    runs = _ROOT / "runs"
    if not (runs / TD_RUN / "config.json").exists():
        pytest.skip(f"smoke run runs/{TD_RUN} not present")
    out_dir = tmp_path_factory.mktemp("models")
    manifest = export_demo.build_manifest(
        td_run=TD_RUN,
        cem_run=CEM_RUN,
        out_dir=out_dir,
        parity_boards=128,
        verbose=False,
    )
    return manifest, out_dir


def test_top_level_shape(built):
    m, _ = built
    assert m["engine_version"] == "1"
    assert isinstance(m["agents"], list)
    assert isinstance(m["selftest"], dict)
    assert isinstance(m["curve"], list)


def test_agent_ids_and_types(built):
    m, _ = built
    by_id = {a["id"]: a for a in m["agents"]}
    assert by_id["random"]["type"] == "random"
    assert by_id["dellacherie"]["type"] == "linear"
    assert by_id["cem"]["type"] == "linear"
    for nn_id in _EXPECTED_NN_IDS:
        assert by_id[nn_id]["type"] == "onnx", nn_id
    for a in m["agents"]:
        assert isinstance(a["id"], str) and isinstance(a["label"], str)


def test_linear_weights_length_8(built):
    m, _ = built
    by_id = {a["id"]: a for a in m["agents"]}
    for lid in ("dellacherie", "cem"):
        w = by_id[lid]["weights"]
        assert len(w) == 8, lid
        assert all(isinstance(x, float) for x in w)


def test_cem_eval_present_and_capped_flagged(built):
    m, _ = built
    cem = next(a for a in m["agents"] if a["id"] == "cem")
    ev = cem["eval"]
    assert isinstance(ev["mean_lines"], (int, float))
    assert "capped" in ev and "games_hit_cap" in ev and "max_pieces" in ev


def test_milestone_onnx_files_exist(built):
    m, out_dir = built
    for a in m["agents"]:
        if a["type"] == "onnx":
            assert "eval" in a
            assert (Path(out_dir) / a["path"]).is_file(), a["path"]


def test_selftest_three_boards_twenty_rows(built):
    m, _ = built
    st = m["selftest"]
    boards = st["boards"]
    assert len(boards) == 3
    for row in boards:
        assert len(row) == 20
        assert all(isinstance(v, int) and 0 <= v < (1 << 10) for v in row)
    vals = st["expected_values"]
    assert len(vals) == 3
    assert all(isinstance(v, float) for v in vals)


def test_curve_nonempty_and_ascending(built):
    m, _ = built
    curve = m["curve"]
    assert len(curve) >= 1
    pts = [c["pieces_trained"] for c in curve]
    assert pts == sorted(pts)
    assert len(pts) == len(set(pts)), "pieces_trained must be strictly ascending"
    for c in curve:
        assert c["eval_median_lines"] is not None
