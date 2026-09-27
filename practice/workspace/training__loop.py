"""训练循环、梯度累积与裁剪：练习骨架，不属于可运行标准实现。"""

from collections.abc import Callable, Iterable
from typing import Protocol
import torch
from torch import Tensor, nn

class OptimizerLike(Protocol):

    def zero_grad(self) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def step(self) -> object:
        raise NotImplementedError('手写练习：请完成核心实现')

def clip_grad_norm(parameters: Iterable[Tensor], max_norm: float) -> Tensor:
    """梯度各为参数形状 [...]；返回裁剪前全局 L2 norm scalar []。"""
    raise NotImplementedError('手写练习：请完成核心实现')

def train_step(model: nn.Module, microbatches: list[tuple[Tensor, Tensor]], optimizer: OptimizerLike, loss_fn: Callable[[Tensor, Tensor], Tensor], max_grad_norm: float | None=None) -> float:
    """每批 x [Bi,...], y [Bi,...]；loss_fn 返回样本平均损失 []。

各批按 Bi/总样本数加权。要求 loss 可按样本分解；BatchNorm、Dropout 等
会令 microbatch 与大 batch 的行为不同，不能据此宣称完全等价。"""
    raise NotImplementedError('手写练习：请完成核心实现')
