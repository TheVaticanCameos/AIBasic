"""二维互相关：循环版与 unfold + matmul 版。限定方形核、groups=1、dilation=1。"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def conv2d_loop(
    x: Tensor, weight: Tensor, bias: Tensor | None = None, stride: int = 1, padding: int = 0
) -> Tensor:
    """x [B,C,H,W], weight [O,C,K,K], bias [O] -> [B,O,Hout,Wout]。"""
    k = weight.shape[-1]
    padded = F.pad(x, (padding, padding, padding, padding))  # [B,C,H+2P,W+2P]
    out_h = (padded.shape[-2] - k) // stride + 1
    out_w = (padded.shape[-1] - k) // stride + 1
    rows: list[Tensor] = []
    for i in range(out_h):
        columns: list[Tensor] = []
        for j in range(out_w):
            patch = padded[:, :, i * stride : i * stride + k, j * stride : j * stride + k]
            value = (patch[:, None] * weight[None]).sum((2, 3, 4))  # [B,O]
            columns.append(value)
        rows.append(torch.stack(columns, dim=-1))  # [B,O,Wout]
    output = torch.stack(rows, dim=-2)  # [B,O,Hout,Wout]
    return output if bias is None else output + bias[None, :, None, None]


def conv2d_unfold(
    x: Tensor, weight: Tensor, bias: Tensor | None = None, stride: int = 1, padding: int = 0
) -> Tensor:
    """同循环版接口；unfold 仅负责提取窗口，卷积计算使用矩阵乘法。"""
    b, _, h, w = x.shape
    out_channels, _, k, _ = weight.shape
    patches = F.unfold(x, kernel_size=k, padding=padding, stride=stride)  # [B,C*K*K,L]
    output = weight.reshape(out_channels, -1) @ patches  # [B,O,L]
    out_h = (h + 2 * padding - k) // stride + 1
    out_w = (w + 2 * padding - k) // stride + 1
    output = output.reshape(b, out_channels, out_h, out_w)
    return output if bias is None else output + bias[None, :, None, None]


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.uniform_(
            self.weight,
            -1 / math.sqrt(in_channels * kernel_size**2),
            1 / math.sqrt(in_channels * kernel_size**2),
        )
        self.stride, self.padding = stride, padding

    def forward(self, x: Tensor) -> Tensor:
        """x [B,C,H,W] -> [B,O,Hout,Wout]。"""
        return conv2d_unfold(x, self.weight, self.bias, self.stride, self.padding)


if __name__ == "__main__":
    print(Conv2d(3, 8, 3, padding=1)(torch.randn(2, 3, 8, 8)).shape)
