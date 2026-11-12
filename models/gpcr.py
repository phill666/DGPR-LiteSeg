"""Frozen GS-HalfConv, renamed GPCR.
Redistribution permission for supplied 130a. GS-HalfConv.py has been confirmed
by the repository maintainer. No additional blanket MIT grant is made here.
Modified: omit demo/unrelated modules, rename class only.
Inactive shuffle, partial widths and scaling are preserved.
See README.md and LICENSE.
"""
import torch
import torch.nn as nn
from torch import Tensor
class HalfConv(nn.Module):
    def __init__(self, dim: int, n_div: int = 2, kernel_size: int = 3):
        super().__init__()
        if n_div < 2:
            raise ValueError("n_div must be at least 2")
        self.dim_conv = dim // n_div
        self.dim_skip = dim - self.dim_conv
        self.partial_conv = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_conv, x_skip = torch.split(
            x, [self.dim_conv, self.dim_skip], dim=1
        )
        return torch.cat((self.partial_conv(x_conv), x_skip), dim=1)


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    b, c, h, w = x.shape
    if groups <= 1 or c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class SEGate(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        weight = self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * weight


class DWConvPW(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.pw(self.dw(x)))


class GPCR(nn.Module):
    """Single-input, shape-preserving gated shuffle half convolution."""

    def __init__(
        self,
        dim: int,
        groups: int = 3,
        n_div: int = 2,
        se_reduction: int = 4,
        shuffle_groups: int = 3,
        dw_kernel: int = 3,
    ):
        super().__init__()
        if groups < 1:
            raise ValueError("groups must be at least 1")
        base = dim // groups
        remainder = dim - base * (groups - 1)
        self.group_dims = [base] * (groups - 1) + [remainder]
        self.group_blocks = nn.ModuleList(
            [HalfConv(d, n_div=n_div, kernel_size=3) for d in self.group_dims]
        )
        self.shuffle_groups = shuffle_groups if dim % shuffle_groups == 0 else 1
        self.mix = DWConvPW(dim, kernel_size=dw_kernel)
        self.se = SEGate(dim, reduction=se_reduction)
        self.res_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        chunks = torch.split(x, self.group_dims, dim=1)
        out = torch.cat(
            [block(chunk) for chunk, block in zip(chunks, self.group_blocks)],
            dim=1,
        )
        out = channel_shuffle(out, self.shuffle_groups)
        out = self.se(self.mix(out))
        return identity + self.res_scale * self.norm(out)
