"""Afterstate ValueNet and board<->tensor encoding (PLAN.md §8).

The ValueNet scores a *post-clear* board (an "afterstate"): input is a binary
``[B, 1, 20, 10]`` grid, output a scalar value per board. Three 3x3 conv layers
(1->16->32->32, pad 1, ReLU) feed a 128-unit FC head to a single value. Grid
only — no hand-crafted features, which is the whole point of the neural agent.

Boards on disk / in the replay buffer are 20 ``uint16`` rows (bit ``c`` set =
column ``c`` filled). ``boards_to_tensor`` unpacks a whole batch of rows to the
``[B, 1, 20, 10]`` float tensor in one vectorized numpy op — no per-cell python
loops on the hot path.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

WIDTH = 10
HEIGHT = 20

# Column bit positions 0..9; a row int's bit c is column c (PLAN.md §2).
_BITS = np.arange(WIDTH, dtype=np.uint16)
_POWERS = (np.uint16(1) << _BITS)  # [1, 2, 4, ..., 512]


def boards_to_array(rows: np.ndarray) -> np.ndarray:
    """(N, 20) uint16 rows -> (N, 20, 10) float32 array of 0/1 cells."""
    rows = np.asarray(rows, dtype=np.uint16)
    return ((rows[..., None] >> _BITS) & 1).astype(np.float32)


def boards_to_tensor(rows: np.ndarray, device: str | torch.device = "cpu") -> torch.Tensor:
    """(N, 20) uint16 rows -> (N, 1, 20, 10) float32 tensor for the ValueNet."""
    arr = boards_to_array(rows)
    t = torch.from_numpy(arr).unsqueeze(1)  # (N, 1, 20, 10)
    if str(device) != "cpu":
        t = t.to(device)
    return t


def array_to_boards(arr: np.ndarray) -> np.ndarray:
    """(N, 20, 10) 0/1 array -> (N, 20) uint16 rows (inverse of boards_to_array)."""
    arr = np.asarray(arr).astype(np.uint16)
    return (arr * _POWERS).sum(axis=-1).astype(np.uint16)


class ValueNet(nn.Module):
    """Afterstate value function V(board) (PLAN.md §8, ~0.83M params)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * HEIGHT * WIDTH, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 20, 10) -> (B,) value per board."""
        return self.head(self.conv(x)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
