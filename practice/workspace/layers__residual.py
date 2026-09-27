"""Residual Block：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn
from layers.batch_norm.implementation import BatchNorm
from layers.conv2d.implementation import Conv2d

class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int=1) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor) -> Tensor:
        """x [B,C,H,W] -> [B,O,ceil(H/S),ceil(W/S)]。"""
        raise NotImplementedError('手写练习：请完成核心实现')
