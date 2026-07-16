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
