"""Adam 与 AdamW：练习骨架，不属于可运行标准实现。"""

from collections.abc import Iterable
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class AdamState:
    step: int
    first: Tensor
    second: Tensor

class Adam:

    def __init__(self, parameters: Iterable[Tensor], lr: float=0.001, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-08, weight_decay: float=0.0, decoupled_weight_decay: bool=False) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def zero_grad(self) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    @torch.no_grad()
    def step(self) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

class AdamW(Adam):

    def __init__(self, parameters: Iterable[Tensor], lr: float=0.001, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-08, weight_decay: float=0.01) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')
