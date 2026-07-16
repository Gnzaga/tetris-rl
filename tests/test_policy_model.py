"""PolicyNet tests (PLAN2.md §6, Phase D).

Covers the frozen architecture (layer shapes, intermediate activation taps,
two- vs three-tuple forward, param count sanity) and the MPS-vs-CPU numeric
parity used by the smoke gate (skipped when MPS is unavailable).
"""

import numpy as np
import pytest
import torch

from tetris.bc import mps_cpu_max_logit_diff, select_device
from tetris.policy_model import (
    ACTIVATION_NAMES,
    NUM_ACTIONS,
    PolicyNet,
    count_parameters,
)


def test_forward_shapes_two_tuple():
    m = PolicyNet()
    x = torch.rand(3, 4, 96, 96)
    logits, value = m(x)
    assert logits.shape == (3, NUM_ACTIONS)
    assert value.shape == (3,)


def test_named_activation_shapes():
    m = PolicyNet()
    x = torch.rand(2, 4, 96, 96)
    logits, value, acts = m(x, return_activations=True)
    assert tuple(acts.keys()) == ACTIVATION_NAMES  # frozen order (ONNX outputs)
    assert acts["conv1"].shape == (2, 16, 23, 23)  # (96-8)//4+1 = 23
    assert acts["conv2"].shape == (2, 32, 10, 10)  # (23-4)//2+1 = 10
    assert acts["conv3"].shape == (2, 32, 8, 8)    # (10-3)//1+1 = 8
    assert acts["fc"].shape == (2, 256)
    assert acts["logits"].shape == (2, NUM_ACTIONS)
    # The tapped logits are the same tensor the two-tuple form returns.
    torch.testing.assert_close(acts["logits"], logits)
    # conv/fc taps are post-ReLU (non-negative).
    for k in ("conv1", "conv2", "conv3", "fc"):
        assert (acts[k] >= 0).all()


def test_param_count_is_stable():
    # Frozen arch + §6 aux target heads (rot 256->4, col 256->10) => fixed
    # parameter count; guards accidental layer edits.
    assert count_parameters(PolicyNet()) == 547_670 + (256 * 4 + 4) + (256 * 10 + 10)


def test_aux_head_shapes_and_isolation():
    # return_aux yields (rot [B,4], col [B,10]) logits; the plain two-tuple
    # forward (inference path) is unchanged by the aux heads' existence.
    m = PolicyNet()
    x = torch.rand(3, 4, 96, 96)
    logits, value, (aux_rot, aux_col) = m(x, return_aux=True)
    assert aux_rot.shape == (3, 4)
    assert aux_col.shape == (3, 10)
    logits2, value2 = m(x)
    torch.testing.assert_close(logits, logits2)
    torch.testing.assert_close(value, value2)


def test_flatten_matches_conv_out():
    m = PolicyNet()
    assert m.fc.in_features == 32 * 8 * 8


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_cpu_logit_parity():
    # Same gate the smoke run enforces: MPS logits match CPU within 1e-3.
    m = PolicyNet()
    diff = mps_cpu_max_logit_diff(m, "mps", seed=1)
    assert diff < 1e-3, f"MPS/CPU logit divergence {diff:.2e}"


def test_select_device_fallback():
    assert select_device("cpu") == "cpu"
    assert select_device("mps") in ("mps", "cpu")
