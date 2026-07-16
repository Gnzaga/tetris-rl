"""Schema checks for the durable model registry (tetris/registry.py) and its
exported indexes (shared/model_registry.json, demo/models/registry.json)."""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tetris import registry as reg  # noqa: E402


def _good_entry() -> dict:
    return reg.make_entry(
        id="x_test", created="2026-07-16T00:00:00+00:00", family="pixel-bc",
        summary="test entry", git_commit="abc123",
        checkpoint="runs/bc_v2/checkpoints/nn_step_0.pt", domain="gray-128",
        eval=reg.make_eval(median_lines=0.0, mean_lines=0.1, pieces_per_game=25.0,
                           presses_per_piece=3.1, press_frac=0.03, thrash=0.0,
                           source="fresh-eval"),
        notes="n",
    )


# -- validate_entry --------------------------------------------------------


def test_good_entry_validates():
    assert reg.validate_entry(_good_entry()) == []


def test_missing_key_flagged():
    e = _good_entry()
    del e["family"]
    assert any("family" in m for m in reg.validate_entry(e))


def test_bad_family_flagged():
    e = _good_entry()
    e["family"] = "not-a-family"
    assert any("family" in m for m in reg.validate_entry(e))


def test_bad_domain_flagged():
    e = _good_entry()
    e["domain"] = "rgb"
    assert any("domain" in m for m in reg.validate_entry(e))


def test_bad_source_flagged():
    e = _good_entry()
    e["eval"]["source"] = "guessed"
    assert any("source" in m for m in reg.validate_entry(e))


def test_bad_created_flagged():
    e = _good_entry()
    e["created"] = "last tuesday"
    assert any("created" in m for m in reg.validate_entry(e))


def test_eval_missing_key_flagged():
    e = _good_entry()
    del e["eval"]["thrash"]
    assert any("thrash" in m for m in reg.validate_entry(e))


# -- families / domains / eval families are the frozen enums ---------------


def test_enums_frozen():
    assert set(reg.FAMILIES) == {"v1-valuenet", "pixel-bc", "pixel-ppo", "mini-experiment"}
    assert set(reg.DOMAINS) == {"camouflage-255", "gray-128", "board-int"}
    assert set(reg.EVAL_SOURCES) == {"historical-metrics", "fresh-eval"}


# -- save / load round-trip (never touches the committed shared copy) ------


def test_save_load_roundtrip(tmp_path):
    entries = [_good_entry()]
    path = tmp_path / "registry.json"
    reg.save(entries, path=path, also_shared=False)
    doc = json.loads(path.read_text())
    assert doc["schema_version"] == reg.SCHEMA_VERSION
    assert reg.load(path)[0]["id"] == "x_test"


def test_save_rejects_invalid(tmp_path):
    bad = _good_entry()
    bad["family"] = "nope"
    with pytest.raises(ValueError):
        reg.save([bad], path=tmp_path / "r.json", also_shared=False)


def test_duplicate_ids_flagged():
    a, b = _good_entry(), _good_entry()
    assert any("duplicate" in m for m in reg.validate([a, b]))


# -- fresh-clone durability (the whole point of the committed index) --------


def test_load_falls_back_to_committed_index(tmp_path):
    """Fresh clone: runs/registry.json absent, shared/model_registry.json
    present -> load() reconstitutes the full lineage from the committed copy."""
    if not reg.SHARED_INDEX.exists():
        pytest.skip("shared index not built — run scripts/registry.py seed")
    canonical = tmp_path / "runs" / "registry.json"   # does not exist
    shared = tmp_path / "shared" / "model_registry.json"
    shared.parent.mkdir(parents=True)
    shared.write_text(reg.SHARED_INDEX.read_text())
    entries = reg.load(path=canonical, shared_path=shared)
    assert len(entries) >= 16, "fallback must return the committed lineage"
    assert reg.validate(entries) == []


def test_register_from_run_merges_onto_committed_lineage(tmp_path, monkeypatch):
    """Fresh-clone auto-register must ADD an entry, never truncate the lineage
    (the original bug: load() ignored the committed index, so the first
    post-clone training run overwrote 16 entries with 1)."""
    if not reg.SHARED_INDEX.exists():
        pytest.skip("shared index not built — run scripts/registry.py seed")
    canonical = tmp_path / "runs" / "registry.json"
    shared = tmp_path / "shared" / "model_registry.json"
    shared.parent.mkdir(parents=True)
    shared.write_text(reg.SHARED_INDEX.read_text())
    n_before = len(json.loads(shared.read_text())["entries"])
    monkeypatch.setattr(reg, "REGISTRY_PATH", canonical)
    monkeypatch.setattr(reg, "SHARED_INDEX", shared)

    # Minimal fake run dir (what register_from_run reads).
    run_dir = tmp_path / "runs" / "bc_test"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "nn_final.pt").write_bytes(b"")
    (run_dir / "config.json").write_text(json.dumps(
        {"created_at": "2026-07-16T20:00:00+00:00", "git_commit": "deadbeef"}))
    (run_dir / "metrics.jsonl").write_text(json.dumps(
        {"eval_median_lines": 0.0, "eval_mean_lines": 0.1,
         "eval_pieces_per_game": 25.0}) + "\n")

    entry = reg.register_from_run(run_dir, family="pixel-bc", domain="gray-128",
                                  ckpt_name="nn_final", summary="fresh-clone test")
    assert entry is not None
    for p in (canonical, shared):
        got = json.loads(p.read_text())["entries"]
        assert len(got) == n_before + 1, f"{p.name}: lineage truncated"
        assert any(e["id"] == "bc_test_final" for e in got)


def test_write_doc_no_churn_when_entries_unchanged(tmp_path):
    """Re-saving an identical entry set must not rewrite the file (no
    `generated`-timestamp churn in the committed index)."""
    path = tmp_path / "r.json"
    entries = [_good_entry()]
    reg.save(entries, path=path, also_shared=False)
    before = path.read_text()
    reg.save(entries, path=path, also_shared=False)
    assert path.read_text() == before


# -- the committed index (shared/model_registry.json) ----------------------


def _load_doc(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not built — run scripts/registry.py seed")
    return json.loads(path.read_text())


def test_shared_index_valid_and_seeded():
    doc = _load_doc(reg.SHARED_INDEX)
    entries = doc["entries"]
    assert reg.validate(entries) == []
    ids = {e["id"] for e in entries}
    # Full lineage present (16 seeded): v1, 4 minis, attempts 1 & 3 (bc+dagger),
    # bc_v2 milestones + dagger, ppo.
    for want in ("v1_td_v1_final", "mini_bc", "mini_gray", "pixel_attempt1_bc",
                 "pixel_attempt3_dagger", "pixel_bc_v2_100", "pixel_bc_v2_dagger1",
                 "pixel_ppo_v2_final"):
        assert want in ids, f"registry missing lineage entry {want}"
    assert len(entries) >= 16
    # Sorted by created date (the Model History ordering).
    created = [e["created"] for e in entries]
    assert created == sorted(created)


def test_demo_index_weights_free():
    doc = _load_doc(reg.DEMO_INDEX)
    entries = doc["entries"]
    assert entries, "demo registry index empty"
    for e in entries:
        # Weights-free: no local checkpoint path leaks into the demo bundle.
        assert "checkpoint" not in e, f"{e['id']} leaks a checkpoint path into the demo"
        for k in ("id", "created", "family", "label", "domain", "eval"):
            assert k in e, f"demo entry {e.get('id')} missing {k}"
        for k in reg._EVAL_KEYS:
            assert k in e["eval"], f"demo entry {e['id']} eval missing {k}"
        assert e["family"] in reg.FAMILIES
        assert e["domain"] in reg.DOMAINS
