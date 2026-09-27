"""SGD 与 Momentum：练习骨架，不属于可运行标准实现。"""

from collections.abc import Iterable
import torch
from torch import Tensor

class SGD:

    def __init__(self, parameters: Iterable[Tensor], lr: float=0.01, momentum: float=0.0, weight_decay: float=0.0) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def zero_grad(self) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    @torch.no_grad()
    def step(self) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')
