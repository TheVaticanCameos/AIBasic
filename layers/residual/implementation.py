"""ResNet basic block：主分支与投影捷径分别承担变换与对齐。"""

import torch
from torch import Tensor, nn

from layers.batch_norm.implementation import BatchNorm
from layers.conv2d.implementation import Conv2d


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.norm1 = BatchNorm(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = BatchNorm(out_channels)
        self.shortcut: nn.Module = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                Conv2d(in_channels, out_channels, 1, stride), BatchNorm(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        """x [B,C,H,W] -> [B,O,ceil(H/S),ceil(W/S)]。"""
        residual = self.shortcut(x)  # [B,O,Hout,Wout]
        hidden = torch.relu(self.norm1(self.conv1(x)))
        return torch.relu(self.norm2(self.conv2(hidden)) + residual)


if __name__ == "__main__":
    print(ResidualBlock(3, 8, 2)(torch.randn(2, 3, 8, 8)).shape)
