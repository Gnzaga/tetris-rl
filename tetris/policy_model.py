"""PolicyNet CNN for the pixel-input keypress agent (PLAN2.md §6, Phase D).

Input is a stack of the last 4 observations, ``[B, 4, 96, 96]`` float in [0, 1]
(see :mod:`tetris.render_obs` / :class:`tetris.bc.BCDataset`). The architecture
is frozen by PLAN2.md §6:

    Conv(4 -> 16, 8x8, stride 4) + ReLU     ->  [B, 16, 23, 23]
    Conv(16 -> 32, 4x4, stride 2) + ReLU    ->  [B, 32, 10, 10]
    Conv(32 -> 32, 3x3, stride 1) + ReLU    ->  [B, 32,  8,  8]
    Flatten                                 ->  [B, 2048]
    FC(2048 -> 256) + ReLU                  ->  [B, 256]
    FC(256 -> 5)  = action logits           ->  [B, 5]
    FC(256 -> 1)  = value head (PPO reuse)  ->  [B]

The value head is unused by BC (only ``logits`` drives the greedy policy) but is
present so Phase E's PPO can reuse the exact same module from scratch.

Named intermediate activations (``conv1``, ``conv2``, ``conv3``, ``fc``,
``logits``) are retrievable via ``forward(..., return_activations=True)`` for the
MarI/O-style demo view; Phase F's ONNX export emits each as a named output. The
forward is written so that export is trivial: one call returns every tap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Frozen I/O dims (PLAN2.md §1/§6).
IN_CHANNELS = 4
OBS_SIZE = 96
NUM_ACTIONS = 5

# Conv output spatial size after the frozen stack (derived, asserted in tests):
#   96 -> (96-8)//4 + 1 = 23 -> (23-4)//2 + 1 = 10 -> (10-3)//1 + 1 = 8
_CONV_OUT = 32 * 8 * 8  # 2048

# Ordered names of the tapped activations (also the ONNX extra-output order).
ACTIVATION_NAMES = ("conv1", "conv2", "conv3", "fc", "logits")


class PolicyNet(nn.Module):
    """Pixel-input keypress policy + value head (module docstring)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(IN_CHANNELS, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.fc = nn.Linear(_CONV_OUT, 256)
        self.pi = nn.Linear(256, NUM_ACTIONS)
        self.v = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        """``[B, 4, 96, 96]`` -> ``(logits [B,5], value [B])``.

        With ``return_activations=True`` also returns an ordered dict of the five
        named taps (``conv1``/``conv2``/``conv3`` post-ReLU feature maps, ``fc``
        post-ReLU 256-vector, ``logits``) for the demo / ONNX export.
        """
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        flat = torch.flatten(c3, 1)
        fc = F.relu(self.fc(flat))
        logits = self.pi(fc)
        value = self.v(fc).squeeze(-1)
        if return_activations:
            acts = {
                "conv1": c1,
                "conv2": c2,
                "conv3": c3,
                "fc": fc,
                "logits": logits,
            }
            return logits, value, acts
        return logits, value


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
