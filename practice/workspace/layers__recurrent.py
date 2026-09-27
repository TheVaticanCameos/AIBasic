"""RNN / LSTM / GRU Cell：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn

class RNNCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """x [B,I], h [B,H] -> h_new [B,H]。"""
        raise NotImplementedError('手写练习：请完成核心实现')

class LSTMCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """x [B,I], state=(h,c)，各 [B,H] -> (h_new,c_new)，各 [B,H]。"""
        raise NotImplementedError('手写练习：请完成核心实现')

class GRUCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """x [B,I], h [B,H] -> h_new [B,H]；reset gate 乘在 hidden 投影之后。"""
        raise NotImplementedError('手写练习：请完成核心实现')
