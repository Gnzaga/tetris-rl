"""demo/models/manifest.json v2 pixel_agents schema check (PLAN2.md §8).

Copies the committed v1 manifest into a tmp dir, runs ``export_demo_v2`` against
the real ``bc_v2`` run (reduced parity board count for speed — the full 1,000-
stack gate is exercised by the script itself), and validates the ``pixel_agents``
section: agent entries, milestone progression, multi-output ONNX files exist, the
FC→action weight matrix is 5×256, the self-test sidecar round-trips, and — the
load-bearing correctness claim — every ONNX output has strict <1e-4 parity with
PyTorch over the REAL rendered-observation domain (not just the random-uint8
gate). v1 manifest keys must survive verbatim. The optional PPO arm is skipped
when its final checkpoint is absent.
"""

import base64
import json
import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import export_demo_v2  # noqa: E402
from tetris.export import (  # noqa: E402
    POLICY_OUTPUT_NAMES,
    load_policynet,
    policy_onnx_outputs,
    policy_torch_outputs,
)

BC_RUN = "bc_v2"
V1_MANIFEST = _ROOT / "demo" / "models" / "manifest.json"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not (_ROOT / "runs" / BC_RUN / "config.json").exists():
        pytest.skip(f"run runs/{BC_RUN} not present")
    if not V1_MANIFEST.exists():
        pytest.skip("v1 manifest not built (run scripts/export_demo.py first)")
    out_dir = tmp_path_factory.mktemp("models_v2")
    # Seed the tmp dir with a v1 manifest so export_demo_v2 has a base to extend.
    manifest_path = out_dir / "manifest.json"
    shutil.copy2(V1_MANIFEST, manifest_path)
    manifest = export_demo_v2.add_pixel_agents(
        manifest_path=manifest_path,
        bc_run=BC_RUN,
        out_dir=out_dir,
        parity_stacks=128,
        verbose=False,
    )
    return manifest, out_dir


def test_v1_keys_preserved(built):
    m, _ = built
    assert m["engine_version"] == "1"
    assert isinstance(m["agents"], list) and len(m["agents"]) >= 1
    assert isinstance(m["selftest"], dict)
    assert isinstance(m["curve"], list)


def test_pixel_block_shape(built):
    m, _ = built
    pa = m["pixel_agents"]
    assert pa["bc_run"] == BC_RUN
    assert pa["ppo_run"] in (None, "ppo_v2")
    assert pa["input_name"] == "obs"
    assert pa["output_names"] == list(POLICY_OUTPUT_NAMES)
    assert pa["activation_outputs"] == ["conv1", "conv2", "conv3", "fc"]
    assert pa["action_legend"] == ["noop", "←", "→", "↑CW", "↓CCW"]
    assert pa["action_names"] == ["noop", "left", "right", "rot_cw", "rot_ccw"]
    assert pa["final"] in {a["id"] for a in pa["agents"]}


def test_milestone_progression_and_files(built):
    m, out_dir = built
    pa = m["pixel_agents"]
    ids = [a["id"] for a in pa["agents"]]
    for expect in ("pixel_nn_000", "pixel_nn_025", "pixel_nn_050", "pixel_nn_100",
                   "pixel_dagger_1"):
        assert expect in ids, expect
    for a in pa["agents"]:
        assert a["type"] == "pixel_onnx"
        assert (Path(out_dir) / a["path"]).is_file(), a["path"]
        ev = a["eval"]
        assert "median_lines" in ev and "mean_lines" in ev and "pieces_per_game" in ev


def test_obs_spec_matches_training(built):
    m, _ = built
    spec = m["pixel_agents"]["obs_spec"]
    assert spec["stack"] == 4
    assert spec["size"] == 96
    assert spec["channels"] == 4
    assert spec["values"] == [0, 128, 255]
    assert spec["active_piece_value"] == 128
    assert "consecutive" in spec["spacing"]


def test_fc_action_weight_matrix(built):
    m, _ = built
    pa = m["pixel_agents"]
    w = pa["fc_action_weight"]
    assert len(w) == 5 and all(len(row) == 256 for row in w)
    assert len(pa["fc_action_bias"]) == 5
    assert all(isinstance(x, float) for x in pa["fc_action_bias"])


def test_selftest_sidecar_roundtrips(built):
    m, out_dir = built
    pa = m["pixel_agents"]
    st = pa["selftest_pixel"]
    sidecar = json.loads((Path(out_dir) / st["path"]).read_text())
    assert sidecar["shape"] == [2, 4, 96, 96]
    raw = np.frombuffer(base64.b64decode(sidecar["stacks_b64"]), dtype=np.uint8)
    assert raw.size == 2 * 4 * 96 * 96
    assert set(np.unique(raw)).issubset({0, 128, 255})
    # Re-run the stored stacks through the final ONNX; logits must match sidecar.
    final = next(a["path"] for a in pa["agents"] if a["id"] == pa["final"])
    stacks = (raw.reshape(2, 4, 96, 96).astype(np.float32) / 255.0)
    logits = policy_onnx_outputs(Path(out_dir) / final, stacks)["logits"]
    exp = np.asarray(sidecar["expected_logits"], dtype=np.float64)
    assert np.max(np.abs(logits - exp)) < 1e-3


def test_strict_parity_on_real_obs_domain(built):
    """Every ONNX output is within a STRICT absolute 1e-4 of PyTorch over the
    real rendered-observation distribution (the random-uint8 gate uses a
    magnitude-aware atol+rtol tolerance for pathologically dense inputs)."""
    from tetris.frame_env import FrameEnv
    from tetris.render_obs import render_env
    import torch

    import json as _json

    m, out_dir = built
    pa = m["pixel_agents"]
    final_path = next(a["path"] for a in pa["agents"] if a["id"] == pa["final"])
    # Resolve the checkpoint backing the FINAL agent (the demo default may be a
    # BC milestone or a DAgger iter — it is the calibrated BC-100 milestone since
    # the toddler-spam fix, not necessarily nn_dagger_1).
    bc_cfg = _json.loads((_ROOT / "runs" / BC_RUN / "config.json").read_text())
    ckpt_map = {f"pixel_nn_{int(p):03d}": info["checkpoint"]
                for p, info in bc_cfg["milestones"].items()}
    ckpt_map["pixel_dagger_0"] = "nn_dagger_0"
    ckpt_map["pixel_dagger_1"] = "nn_dagger_1"
    ckpt = _ROOT / "runs" / BC_RUN / "checkpoints" / f"{ckpt_map[pa['final']]}.pt"
    model = load_policynet(ckpt)

    stacks = []
    for seed in range(30):
        env = FrameEnv(seed=seed)
        h = deque(maxlen=4)
        n = 0
        while not env.game_over and n < 30:
            if env.is_decision_tick:
                o = render_env(env)
                if not h:
                    for _ in range(4):
                        h.append(o)
                else:
                    h.append(o)
                stacks.append(np.stack(h).astype(np.float32) / 255.0)
                n += 1
                with torch.no_grad():
                    a = int(model(torch.from_numpy(stacks[-1][None]))[0].argmax())
                env.apply_action(a)
            env.tick()
        if len(stacks) >= 500:
            break
    S = np.stack(stacks[:500])
    t = policy_torch_outputs(model, S)
    o = policy_onnx_outputs(Path(out_dir) / final_path, S)
    for name in POLICY_OUTPUT_NAMES:
        d = float(np.max(np.abs(t[name] - o[name])))
        assert d < 1e-4, f"{name} real-obs parity {d:.2e} >= 1e-4"
