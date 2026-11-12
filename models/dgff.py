"""Frozen DFF core, renamed DGFF.
Supplied source 66a. DFF_2D.py credits Bilibili creator CV缝合救星.
Redistribution permission has been confirmed by the repository maintainer.
No additional blanket MIT grant is made by this notice.
Preserve attribution. See README.md and LICENSE.
Modified: retain frozen core/shape check and rename class only.
"""
import torch
import torch.nn as nn
from torch import Tensor
class DGFF(nn.Module):
    """Two-input dynamic fusion for equal-shaped NCHW tensors."""

    def __init__(self, dim: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        if x.shape != skip.shape:
            raise ValueError("DFF inputs must have the same NCHW shape")
        output = torch.cat([x, skip], dim=1)
        output = output * self.conv_atten(self.avg_pool(output))
        output = self.conv_redu(output)
        att = self.nonlin(self.conv1(x) + self.conv2(skip))
        return output * att
