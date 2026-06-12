from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from filters.grayscale import rgb_to_luminance


FILTER_NAME = "sobel"
name = FILTER_NAME


class SobelInputFilter(nn.Module):
    name = FILTER_NAME

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "kernel_x",
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "kernel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
            ).view(1, 1, 3, 3),
        )
        self.scale = 1.0 / (4.0 * math.sqrt(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = rgb_to_luminance(x)
        padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        gradient_x = F.conv2d(padded, self.kernel_x)
        gradient_y = F.conv2d(padded, self.kernel_y)
        magnitude = torch.sqrt(gradient_x.square() + gradient_y.square())
        edges = (magnitude * self.scale).clamp(0.0, 1.0)
        return edges.expand(-1, 3, -1, -1)


def create_filter() -> nn.Module:
    return SobelInputFilter()
