"""目标网络同步与参数冻结：练习骨架，不属于可运行标准实现。"""

from collections.abc import Iterator
from contextlib import contextmanager
import torch
from torch import nn

@torch.no_grad()
def hard_update(target: nn.Module, source: nn.Module) -> None:
    """复制参数和 buffers；目标与源必须同结构。"""
    raise NotImplementedError('手写练习：请完成核心实现')

@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float=0.005) -> None:
    """每个同形状参数 [...]：target <- (1-tau)*target + tau*source。

floating buffers 同样插值，整数 buffers 直接复制。"""
    raise NotImplementedError('手写练习：请完成核心实现')

@contextmanager
def frozen_parameters(*modules: nn.Module) -> Iterator[None]:
    """临时关闭参数梯度，但保留对网络输入（如 actor 动作）的梯度。"""
    raise NotImplementedError('手写练习：请完成核心实现')
