"""Autograd 与自定义 backward：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

class Square(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        """x [...] -> y [...]；保存 backward 需要的输入。"""
        # raise NotImplementedError('手写练习：请完成核心实现')
        ctx.save_for_backward(x)
        return x.square()

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        """grad_output [...] -> grad_x [...]；链式法则。"""
        # raise NotImplementedError('手写练习：请完成核心实现')
        (x, ) = ctx.saved_tensors
        return 2 * grad_output * x 

def square(x: Tensor) -> Tensor:
    """x floating [...] -> [...]，使用自定义 autograd Function。"""
    # raise NotImplementedError('手写练习：请完成核心实现')
    return Square.apply(x)
