from __future__ import annotations

import torch
from torch import nn


FILTER_NAME = "grayscale"
name = FILTER_NAME


class GrayscaleInputFilter(nn.Module):
    name = FILTER_NAME

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = rgb_to_luminance(x)
        return gray.expand(-1, 3, -1, -1)


def create_filter() -> nn.Module:
    return GrayscaleInputFilter()


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(
            "grayscale input filter expects a BCHW tensor with 3 channels"
        )

    weights = x.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)
