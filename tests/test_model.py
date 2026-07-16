"""ValueNet architecture + board<->tensor encoding (PLAN.md §8)."""

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tetris.model import (
    ValueNet,
    array_to_boards,
    boards_to_array,
    boards_to_tensor,
    count_parameters,
)


def test_param_count_is_about_0_9m():
    n = count_parameters(ValueNet())
    # Architecture is fixed by §8; ~0.83M params (documented "~0.9M").
    assert 800_000 <= n <= 950_000, n


def test_forward_shape_is_flat_value_vector():
    model = ValueNet()
    x = torch.zeros((7, 1, 20, 10))
    out = model(x)
    assert out.shape == (7,)


def test_boards_to_tensor_shape_and_binary():
    rows = np.array([[0, 1023, 5] + [0] * 17], dtype=np.uint16)  # (1, 20)
    t = boards_to_tensor(rows)
    assert t.shape == (1, 1, 20, 10)
    assert set(np.unique(t.numpy()).tolist()) <= {0.0, 1.0}
    # Row value 5 = 0b0000000101 -> columns 0 and 2 filled.
    assert t[0, 0, 2, 0].item() == 1.0
    assert t[0, 0, 2, 2].item() == 1.0
    assert t[0, 0, 2, 1].item() == 0.0


def test_encode_decode_round_trip():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 1024, size=(50, 20)).astype(np.uint16)
    arr = boards_to_array(rows)
    back = array_to_boards(arr)
    assert np.array_equal(back, rows)


def test_forward_is_deterministic_and_batch_independent():
    torch.manual_seed(1)
    model = ValueNet().eval()
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 1024, size=(16, 20)).astype(np.uint16)
    with torch.no_grad():
        batched = model(boards_to_tensor(rows)).numpy()
        singles = np.array(
            [model(boards_to_tensor(rows[i : i + 1])).item() for i in range(16)]
        )
    assert np.allclose(batched, singles, atol=1e-5)
