"""Extend ``demo/models/manifest.json`` with the v2 pixel agents (PLAN2.md §8).

This is the v2 companion to :mod:`export_demo`. It does NOT touch the v1 export
path: v1's ``export_demo.py`` stays untouched and MUST be run FIRST (it builds the
base ``manifest.json`` + vendors onnxruntime-web). This script then READS that
existing manifest, exports the ``bc_v2`` pixel-policy milestones (and ``ppo_v2``'s
final checkpoint, if present) to MULTI-OUTPUT ONNX (opset 17, dynamic batch —
logits + conv1/conv2/conv3/fc taps + aux rot/col heads), gates each on the
<1e-4 activation parity check, and ADDS/REPLACES a single ``pixel_agents`` section
in the manifest. Every v1 manifest key is preserved verbatim, so the v1 demo keeps
working against the extended file.

Ordering requirement (documented, enforced): run ``export_demo.py`` first so a v1
manifest exists at ``--manifest``; this script fails hard if it is missing.

    python scripts/export_demo.py --td-run td_v1 --cem-run cem_v1   # FIRST (v1)
    python scripts/export_demo_v2.py --bc-run bc_v2                 # THEN (v2)

The ``ppo_v2`` arm is optional: it is included only if its final checkpoint
(``runs/<ppo-run>/checkpoints/nn_final.pt``) already exists, else omitted (the
PPO trainer may still be running its time-box). Nothing else in the manifest —
or the v1 demo — depends on it.

Obs/stack semantics baked into the manifest (must match tetris/bc.py and the
demo's pixel_agent.js exactly): a stack of the last 4 CONSECUTIVE decision-tick
observations, oldest→newest ``[t-3, t-2, t-1, t]`` (clamped by repeating the
first frame at episode start), each 96×96 uint8 in {0,128,255} normalized /255.
"""

import _pathshim  # noqa: F401
import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tetris.export import (
    PARITY_BOARDS,
    PARITY_TOL,
    POLICY_INPUT_NAME,
    POLICY_OUTPUT_NAMES,
    POLICY_OUTPUT_SHAPES,
    export_policy_checkpoint,
    load_policynet,
    policy_onnx_outputs,
    random_obs_stacks,
)

_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = _ROOT / "runs"
DEMO_DIR = _ROOT / "demo"
DEFAULT_MODELS = DEMO_DIR / "models"

# Honest action legend / names (PLAN2.md §1/§8).
ACTION_LEGEND = ["noop", "←", "→", "↑CW", "↓CCW"]
ACTION_NAMES = ["noop", "left", "right", "rot_cw", "rot_ccw"]

# Activation taps the demo animates (logits/aux are handled separately).
ACTIVATION_OUTPUTS = ["conv1", "conv2", "conv3", "fc"]

SELFTEST_SIDECAR = "pixel_selftest.json"
SELFTEST_SEED = 20260716
SELFTEST_TOL = 1e-3

# BC milestone percent -> honest label. All best-effort: median 0 lines,
# ~25-piece games (Phase D gate NOT MET, re-scoped — PLAN2.md §6).
_BC_LABELS = {
    0: "PixelNet — untrained (0%)",
    25: "PixelNet — behavior-cloned 25%",
    50: "PixelNet — behavior-cloned 50%",
    100: "PixelNet — behavior-cloned (BC final; best-effort, ~25 pieces)",
}


def _read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return rows
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _nearest_eval(metrics: list[dict[str, Any]], step: int,
                  median_override: float | None = None) -> dict[str, Any]:
    """Eval stats for the metrics row whose pieces_trained is nearest ``step``.

    For BC/DAgger ``pieces_trained`` carries the optimizer-step index. When a
    checkpoint payload already knows its closed-loop median, pass it as
    ``median_override`` so the manifest reflects the exact per-checkpoint number.
    """
    evals = [m for m in metrics if m.get("eval_median_lines") is not None]
    row = (min(evals, key=lambda m: abs((m.get("pieces_trained") or 0) - step))
           if evals else {})
    median = median_override if median_override is not None else row.get("eval_median_lines")
    return {
        "median_lines": median,
        "mean_lines": row.get("eval_mean_lines"),
        "pieces_per_game": row.get("eval_pieces_per_game"),
        "optimizer_step": step,
    }


def _ckpt_meta(ckpt_path: Path) -> dict[str, Any]:
    """Read a PolicyNet checkpoint's non-tensor metadata (optimizer_step, eval)."""
    import torch

    ck = torch.load(str(ckpt_path), map_location="cpu")
    return {
        "optimizer_step": int(ck.get("optimizer_step", 0)) if isinstance(ck, dict) else 0,
        "eval_median_lines": ck.get("eval_median_lines") if isinstance(ck, dict) else None,
    }


def _pixel_entry(pid: str, label: str, onnx_name: str,
                 eval_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": pid,
        "type": "pixel_onnx",
        "label": label,
        "path": onnx_name,
        "eval": eval_stats,
    }


def _obs_spec() -> dict[str, Any]:
    return {
        "stack": 4,
        "stack_order": "oldest_to_newest [t-3, t-2, t-1, t]",
        "spacing": "consecutive decision ticks (start padded by repeating t)",
        "size": 96,
        "channels": 4,
        "values": [0, 128, 255],
        "normalize": "x / 255",
        "active_piece_value": 128,
        "tick_hz": 30,
        "decision_hz": 10,
    }


def _write_selftest_sidecar(final_onnx: Path, out_dir: Path) -> dict[str, Any]:
    """2 fixed obs stacks -> final pixel ONNX logits (onnxruntime). Sidecar file.

    The stacks (2×4×96×96 uint8) are stored base64-packed in a sidecar to keep
    manifest.json lean; the manifest references the sidecar path + tolerance.
    """
    raw = (random_obs_stacks(2, SELFTEST_SEED) * 255.0).round().astype(np.uint8)
    stacks = raw.astype(np.float32) / 255.0
    logits = policy_onnx_outputs(final_onnx, stacks)["logits"]
    sidecar = {
        "shape": list(raw.shape),
        "dtype": "uint8",
        "stacks_b64": base64.b64encode(raw.tobytes()).decode("ascii"),
        "expected_logits": [[float(v) for v in row] for row in logits],
        "tol": SELFTEST_TOL,
    }
    with open(out_dir / SELFTEST_SIDECAR, "w") as f:
        json.dump(sidecar, f)
        f.write("\n")
    return {"path": SELFTEST_SIDECAR, "tol": SELFTEST_TOL,
            "expected_logits": sidecar["expected_logits"]}


def add_pixel_agents(
    manifest_path: Path = DEFAULT_MODELS / "manifest.json",
    bc_run: str = "bc_v2",
    ppo_run: str = "ppo_v2",
    runs_dir: Path = RUNS_DIR,
    out_dir: Path | None = None,
    parity_stacks: int = PARITY_BOARDS,
    parity_tol: float = PARITY_TOL,
    verbose: bool = True,
) -> dict[str, Any]:
    """Read the v1 manifest, add a ``pixel_agents`` section, write it back.

    Fails hard if the v1 manifest is missing (run ``export_demo.py`` first) or if
    any pixel ONNX export misses the activation parity gate.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} not found — run scripts/export_demo.py FIRST to "
            f"build the v1 manifest, then re-run export_demo_v2.py."
        )
    out_dir = Path(out_dir) if out_dir is not None else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(manifest_path)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    bc_dir = runs_dir / bc_run
    bc_config = _read_json(bc_dir / "config.json")
    milestones = bc_config.get("milestones") or {}
    if not milestones:
        raise SystemExit(f"{bc_dir}/config.json has no 'milestones' block")
    bc_metrics = _read_metrics(bc_dir)

    agents: list[dict[str, Any]] = []
    final_onnx: Path | None = None

    # BC milestones (0/25/50/100%).
    for pct in sorted(int(p) for p in milestones):
        info = milestones[str(pct)]
        ckpt = bc_dir / "checkpoints" / f"{info['checkpoint']}.pt"
        pid = f"pixel_nn_{pct:03d}"
        onnx_name = f"{pid}.onnx"
        rep = export_policy_checkpoint(ckpt, out_dir / onnx_name,
                                       n_stacks=parity_stacks, tol=parity_tol)
        meta = _ckpt_meta(ckpt)
        _log(f"  {pid}: {info['checkpoint']}.pt -> {onnx_name} "
             f"(parity max|d|={rep['max_abs_diff']:.2e})")
        agents.append(_pixel_entry(
            pid, _BC_LABELS.get(pct, f"PixelNet — {pct}% trained"), onnx_name,
            _nearest_eval(bc_metrics, meta["optimizer_step"], meta["eval_median_lines"]),
        ))

    # DAgger iterations (dagger_0, dagger_1=final shipped agent).
    dagger_labels = {
        0: "PixelNet — DAgger iter 1 (best-effort)",
        1: "PixelNet — DAgger final (shipped; best-effort, median 0 lines, ~25 pieces)",
    }
    final_pixel_id = None
    for k in (0, 1):
        ckpt = bc_dir / "checkpoints" / f"nn_dagger_{k}.pt"
        if not ckpt.exists():
            continue
        pid = f"pixel_dagger_{k}"
        onnx_name = f"{pid}.onnx"
        rep = export_policy_checkpoint(ckpt, out_dir / onnx_name,
                                       n_stacks=parity_stacks, tol=parity_tol)
        meta = _ckpt_meta(ckpt)
        _log(f"  {pid}: nn_dagger_{k}.pt -> {onnx_name} "
             f"(parity max|d|={rep['max_abs_diff']:.2e})")
        agents.append(_pixel_entry(
            pid, dagger_labels[k], onnx_name,
            _nearest_eval(bc_metrics, meta["optimizer_step"], meta["eval_median_lines"]),
        ))
        final_onnx = out_dir / onnx_name
        final_pixel_id = pid

    if final_onnx is None:
        # No DAgger checkpoints: fall back to the 100% BC milestone as final.
        final_pixel_id = "pixel_nn_100"
        final_onnx = out_dir / "pixel_nn_100.onnx"

    # Optional PPO arm — only if its final checkpoint already exists.
    ppo_included = False
    ppo_final = runs_dir / ppo_run / "checkpoints" / "nn_final.pt"
    if ppo_final.exists():
        pid = "pixel_ppo_final"
        onnx_name = f"{pid}.onnx"
        rep = export_policy_checkpoint(ppo_final, out_dir / onnx_name,
                                       n_stacks=parity_stacks, tol=parity_tol)
        ppo_metrics = _read_metrics(runs_dir / ppo_run)
        meta = _ckpt_meta(ppo_final)
        _log(f"  {pid}: nn_final.pt -> {onnx_name} "
             f"(parity max|d|={rep['max_abs_diff']:.2e})")
        agents.append(_pixel_entry(
            pid, "PixelNet — pure PPO (time-boxed contrast)", onnx_name,
            _nearest_eval(ppo_metrics, meta["optimizer_step"], meta["eval_median_lines"]),
        ))
        ppo_included = True
    else:
        _log(f"  (ppo arm omitted — {ppo_final} not present yet)")

    # FC->action weight matrix (5×256) + bias from the FINAL model, for the demo
    # node-wire graph (edge brightness = |weight × activation|).
    dagger1 = bc_dir / "checkpoints" / "nn_dagger_1.pt"
    final_ckpt = dagger1 if dagger1.exists() else (
        bc_dir / "checkpoints" / f"{milestones['100']['checkpoint']}.pt")
    final_model = load_policynet(final_ckpt)
    fc_w = final_model.pi.weight.detach().cpu().numpy().astype(np.float64)  # (5,256)
    fc_b = final_model.pi.bias.detach().cpu().numpy().astype(np.float64)    # (5,)

    selftest = _write_selftest_sidecar(final_onnx, out_dir)

    manifest["pixel_agents"] = {
        "bc_run": bc_run,
        "ppo_run": ppo_run if ppo_included else None,
        "final": final_pixel_id,
        "input_name": POLICY_INPUT_NAME,
        "output_names": list(POLICY_OUTPUT_NAMES),
        "output_shapes": POLICY_OUTPUT_SHAPES,
        "activation_outputs": ACTIVATION_OUTPUTS,
        "action_legend": ACTION_LEGEND,
        "action_names": ACTION_NAMES,
        "obs_spec": _obs_spec(),
        "fc_action_weight": [[float(v) for v in row] for row in fc_w],
        "fc_action_bias": [float(v) for v in fc_b],
        "selftest_pixel": selftest,
        "agents": agents,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    _log(f"wrote {manifest_path} (+pixel_agents: {len(agents)} agents, "
         f"final={final_pixel_id}, ppo={'yes' if ppo_included else 'no'})")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Add v2 pixel agents to demo manifest")
    ap.add_argument("--bc-run", default="bc_v2", help="BC+DAgger run name under runs/")
    ap.add_argument("--ppo-run", default="ppo_v2", help="PPO run name (optional/if-present)")
    ap.add_argument("--manifest", default=str(DEFAULT_MODELS / "manifest.json"),
                    help="existing v1 manifest to extend")
    ap.add_argument("--parity-stacks", type=int, default=PARITY_BOARDS)
    args = ap.parse_args(argv)

    print(f"extending {args.manifest} with pixel agents from bc_run={args.bc_run}")
    add_pixel_agents(
        manifest_path=Path(args.manifest),
        bc_run=args.bc_run,
        ppo_run=args.ppo_run,
        parity_stacks=args.parity_stacks,
    )
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
