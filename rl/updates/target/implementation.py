"""目标网络同步，以及 actor 更新时冻结 critic 参数的上下文。"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


@torch.no_grad()
def hard_update(target: nn.Module, source: nn.Module) -> None:
    """复制参数和 buffers；目标与源必须同结构。"""
    target.load_state_dict(source.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005) -> None:
    """每个同形状参数 [...]：target <- (1-tau)*target + tau*source。

    floating buffers 同样插值，整数 buffers 直接复制。
    """
    if not 0 <= tau <= 1:
        raise ValueError("tau must be in [0,1]")
    target_state, source_state = target.state_dict(), source.state_dict()
    if target_state.keys() != source_state.keys():
        raise ValueError("source and target structures must match")
    for name, target_tensor in target_state.items():
        source_tensor = source_state[name]
        if target_tensor.is_floating_point():
            target_tensor.lerp_(source_tensor, tau)
        else:
            target_tensor.copy_(source_tensor)


@contextmanager
def frozen_parameters(*modules: nn.Module) -> Iterator[None]:
    """临时关闭参数梯度，但保留对网络输入（如 actor 动作）的梯度。"""
    parameters = [p for module in modules for p in module.parameters()]
    original = [p.requires_grad for p in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, original):
            parameter.requires_grad_(flag)


if __name__ == "__main__":
    source, target = nn.Linear(2, 1), nn.Linear(2, 1)
    hard_update(target, source)
    soft_update(target, source)
    print(torch.equal(source.weight, target.weight))
