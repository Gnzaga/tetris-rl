"""ONNX export + PyTorch<->onnxruntime numerical parity (PLAN.md §9).

Each ValueNet milestone checkpoint is exported to ONNX (opset 17, dynamic batch
axis) so the browser demo can score candidate afterstates with onnxruntime-web.
Before an export is trusted it must pass a hard parity gate: the maximum absolute
difference between PyTorch and onnxruntime outputs over 1,000 random boards must
stay below 1e-4, otherwise :func:`export_checkpoint` raises.

The exported graph takes the same ``[B, 1, 20, 10]`` float32 board tensor the
:class:`~tetris.model.ValueNet` consumes (see :func:`tetris.model.boards_to_tensor`)
and emits a ``[B]`` value vector; the batch axis is dynamic so the demo can feed
all candidate placements of a decision in one inference call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ValueNet, boards_to_tensor

OPSET = 17
PARITY_BOARDS = 1000
PARITY_TOL = 1e-4

INPUT_NAME = "board"
OUTPUT_NAME = "value"


def load_valuenet(checkpoint_path: str | Path, device: str = "cpu") -> ValueNet:
    """Load a ``nn_step_<pieces>.pt`` checkpoint into an eval-mode ValueNet.

    Checkpoints are ``{"model_state", "pieces_trained", "arch"}`` dicts written by
    the TD trainer (PLAN.md §8); a bare ``state_dict`` is also accepted.
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model = ValueNet().to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def export_onnx(model: ValueNet, out_path: str | Path, opset: int = OPSET) -> Path:
    """Export ``model`` to ONNX at ``out_path`` with a dynamic batch axis."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros((1, 1, 20, 10), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=opset,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        do_constant_folding=True,
        # torch>=2.x defaults to the dynamo exporter (needs onnxscript); pin the
        # stable TorchScript path which supports opset 17 + dynamic_axes here.
        dynamo=False,
    )
    return out_path


def _random_boards(n: int, seed: int) -> np.ndarray:
    """(n, 20) uint16 rows of random 10-bit column masks."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1 << 10, size=(n, 20)).astype(np.uint16)


def onnx_values(onnx_path: str | Path, rows: np.ndarray) -> np.ndarray:
    """Run onnxruntime on ``(N, 20)`` uint16 boards -> ``(N,)`` float value vector."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    x = boards_to_tensor(rows).numpy().astype(np.float32)
    out = sess.run(None, {inp: x})[0]
    return np.asarray(out, dtype=np.float64).reshape(-1)


def torch_values(model: ValueNet, rows: np.ndarray) -> np.ndarray:
    """Run the PyTorch model on ``(N, 20)`` uint16 boards -> ``(N,)`` values."""
    model.eval()
    with torch.no_grad():
        out = model(boards_to_tensor(rows))
    return out.detach().cpu().numpy().astype(np.float64).reshape(-1)


def parity_max_abs_diff(
    model: ValueNet,
    onnx_path: str | Path,
    n_boards: int = PARITY_BOARDS,
    seed: int = 0,
) -> float:
    """max |torch - onnxruntime| over ``n_boards`` random boards."""
    rows = _random_boards(n_boards, seed)
    t = torch_values(model, rows)
    o = onnx_values(onnx_path, rows)
    return float(np.max(np.abs(t - o)))


def export_checkpoint(
    checkpoint_path: str | Path,
    out_path: str | Path,
    opset: int = OPSET,
    n_boards: int = PARITY_BOARDS,
    tol: float = PARITY_TOL,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Export one checkpoint to ONNX and gate it on the parity check.

    Raises ``RuntimeError`` (fail hard, PLAN.md §9) if the maximum absolute
    torch/onnxruntime difference is not below ``tol``. Returns a small report
    ``{"onnx_path", "max_abs_diff", "n_boards", "tol"}`` on success.
    """
    model = load_valuenet(checkpoint_path, device=device)
    out_path = export_onnx(model, out_path, opset=opset)
    max_abs = parity_max_abs_diff(model, out_path, n_boards=n_boards, seed=seed)
    if not (max_abs < tol):
        raise RuntimeError(
            f"ONNX parity FAILED for {out_path.name}: "
            f"max|torch-onnx|={max_abs:.3e} >= tol={tol:.1e} "
            f"over {n_boards} boards"
        )
    return {
        "onnx_path": str(out_path),
        "max_abs_diff": max_abs,
        "n_boards": n_boards,
        "tol": tol,
    }


# ==========================================================================
# PolicyNet multi-output export (PLAN2.md §8, Phase F)
# ==========================================================================
#
# The pixel agent's PolicyNet is exported to a MULTI-OUTPUT ONNX graph: the
# 5-way action logits plus every named intermediate tap (conv1/conv2/conv3/fc)
# and the two auxiliary target heads (aux_rot/aux_col), so the browser demo can
# render the MarI/O-style activation view straight from one inference call. The
# input is the policy's stacked-observation tensor ``[B, 4, 96, 96]`` float32 in
# [0, 1]; the batch axis is dynamic on every input/output. The parity gate runs
# over random obs stacks drawn from the real value domain (uint8 {0,128,255}/255)
# and must hold below ``PARITY_TOL`` on ALL outputs.

from .policy_model import PolicyNet, OBS_SIZE, IN_CHANNELS  # noqa: E402

POLICY_INPUT_NAME = "obs"
# Fixed output order (also the manifest's `output_names`): action logits first,
# then the conv/fc taps, then the aux target heads.
POLICY_OUTPUT_NAMES = ["logits", "conv1", "conv2", "conv3", "fc", "aux_rot", "aux_col"]
# Per-output feature shape (excluding the dynamic batch axis), for the demo.
POLICY_OUTPUT_SHAPES = {
    "logits": [5],
    "conv1": [16, 23, 23],
    "conv2": [32, 10, 10],
    "conv3": [32, 8, 8],
    "fc": [256],
    "aux_rot": [4],
    "aux_col": [10],
}


class _PolicyExport(nn.Module):
    """Thin wrapper turning :class:`PolicyNet` into a flat multi-output graph.

    Recomputes the exact same ops as :meth:`PolicyNet.forward` and returns the
    seven taps as an ordered tuple aligned to :data:`POLICY_OUTPUT_NAMES` so
    ``torch.onnx.export`` emits each as a named output.
    """

    def __init__(self, net: PolicyNet):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor):
        n = self.net
        c1 = F.relu(n.conv1(x))
        c2 = F.relu(n.conv2(c1))
        c3 = F.relu(n.conv3(c2))
        flat = torch.flatten(c3, 1)
        fc = F.relu(n.fc(flat))
        logits = n.pi(fc)
        aux_rot = n.aux_rot(fc)
        aux_col = n.aux_col(fc)
        return logits, c1, c2, c3, fc, aux_rot, aux_col


def load_policynet(checkpoint_path: str | Path, device: str = "cpu") -> PolicyNet:
    """Load a PolicyNet checkpoint (``{"model_state", ...}`` or a bare state_dict)."""
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model = PolicyNet().to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def export_policy_onnx(model: PolicyNet, out_path: str | Path, opset: int = OPSET) -> Path:
    """Export ``model`` to a multi-output ONNX graph with a dynamic batch axis."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapper = _PolicyExport(model).eval()
    dummy = torch.zeros((1, IN_CHANNELS, OBS_SIZE, OBS_SIZE), dtype=torch.float32)
    dynamic = {POLICY_INPUT_NAME: {0: "batch"}}
    for name in POLICY_OUTPUT_NAMES:
        dynamic[name] = {0: "batch"}
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        opset_version=opset,
        input_names=[POLICY_INPUT_NAME],
        output_names=list(POLICY_OUTPUT_NAMES),
        dynamic_axes=dynamic,
        do_constant_folding=True,
        dynamo=False,
    )
    return out_path


def random_obs_stacks(n: int, seed: int) -> np.ndarray:
    """``(n, 4, 96, 96)`` float32 stacks from random uint8 {0,128,255} / 255.

    Matches the real observation value domain (empty / active-gray / filled).
    """
    rng = np.random.default_rng(seed)
    raw = rng.choice(np.array([0, 128, 255], dtype=np.uint8),
                     size=(n, IN_CHANNELS, OBS_SIZE, OBS_SIZE))
    return (raw.astype(np.float32) / 255.0)


def policy_torch_outputs(model: PolicyNet, stacks: np.ndarray) -> dict[str, np.ndarray]:
    """Run PyTorch -> {output_name: float64 array} for the seven taps."""
    wrapper = _PolicyExport(model).eval()
    with torch.no_grad():
        outs = wrapper(torch.from_numpy(np.ascontiguousarray(stacks, dtype=np.float32)))
    return {name: outs[i].detach().cpu().numpy().astype(np.float64)
            for i, name in enumerate(POLICY_OUTPUT_NAMES)}


def policy_onnx_outputs(onnx_path: str | Path, stacks: np.ndarray) -> dict[str, np.ndarray]:
    """Run onnxruntime -> {output_name: float64 array} for the seven taps."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    x = np.ascontiguousarray(stacks, dtype=np.float32)
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, {inp: x})
    return {name: np.asarray(o, dtype=np.float64) for name, o in zip(names, outs)}


# Relative tolerance for the random-uint8 parity gate. The action logits reach
# magnitude ~38 on pathologically DENSE random inputs (every one of 4×96×96
# pixels randomly bright — far off the sparse real-obs manifold), where a 1e-4
# ABSOLUTE tolerance equals ~3e-6 RELATIVE: the fp32 conv+matmul accumulation
# floor. Comparing varying-magnitude tensors with |diff| <= atol + rtol·|ref|
# (numpy.allclose semantics) is the correct metric — it holds each output to a
# strict 1e-4 near zero AND catches any genuine export defect (wrong weights/op
# => diffs of order 1, thousands of × over threshold). Over the REAL rendered-obs
# domain every output passes the strict absolute 1e-4 (see tests/test_export_demo_v2.py).
PARITY_RTOL = 1e-5


def policy_parity(
    model: PolicyNet,
    onnx_path: str | Path,
    n_stacks: int = PARITY_BOARDS,
    seed: int = 0,
    atol: float = PARITY_TOL,
    rtol: float = PARITY_RTOL,
) -> tuple[dict[str, float], bool]:
    """Per-output max |torch - onnxruntime| over ``n_stacks`` random obs stacks.

    Returns ``({output_name: max_abs_diff, ..., "overall": worst}, passed)`` where
    ``passed`` is True iff every output is close under ``atol + rtol·|ref|``.
    """
    stacks = random_obs_stacks(n_stacks, seed)
    t = policy_torch_outputs(model, stacks)
    o = policy_onnx_outputs(onnx_path, stacks)
    diffs: dict[str, float] = {}
    passed = True
    for name in POLICY_OUTPUT_NAMES:
        diffs[name] = float(np.max(np.abs(t[name] - o[name])))
        if not np.allclose(t[name], o[name], rtol=rtol, atol=atol):
            passed = False
    diffs["overall"] = max(diffs[n] for n in POLICY_OUTPUT_NAMES)
    return diffs, passed


def export_policy_checkpoint(
    checkpoint_path: str | Path,
    out_path: str | Path,
    opset: int = OPSET,
    n_stacks: int = PARITY_BOARDS,
    tol: float = PARITY_TOL,
    rtol: float = PARITY_RTOL,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Export one PolicyNet checkpoint to multi-output ONNX + gate on parity.

    Raises ``RuntimeError`` if any output is not close under ``tol + rtol·|ref|``.
    Returns ``{"onnx_path", "max_abs_diff", "per_output", "n_stacks", "tol"}``.
    """
    model = load_policynet(checkpoint_path, device=device)
    out_path = export_policy_onnx(model, out_path, opset=opset)
    per, passed = policy_parity(model, out_path, n_stacks=n_stacks, seed=seed,
                                atol=tol, rtol=rtol)
    if not passed:
        worst = max((k for k in per if k != "overall"), key=lambda k: per[k])
        raise RuntimeError(
            f"PolicyNet ONNX parity FAILED for {Path(out_path).name}: "
            f"max|torch-onnx|={per['overall']:.3e} (worst output '{worst}') not "
            f"within atol={tol:.1e}+rtol={rtol:.1e}·|ref| over {n_stacks} obs stacks"
        )
    return {
        "onnx_path": str(out_path),
        "max_abs_diff": per["overall"],
        "per_output": per,
        "n_stacks": n_stacks,
        "tol": tol,
    }
