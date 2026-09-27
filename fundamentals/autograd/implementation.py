"""自定义 backward 与梯度累积示例。"""

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx


class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        """x [...] -> y [...]；保存 backward 需要的输入。"""
        ctx.save_for_backward(x)
        return x.square()

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        """grad_output [...] -> grad_x [...]；链式法则。"""
        # saved_tensors 由 autograd 在运行时注入，FunctionCtx 的静态 stub 未声明它。
        (x,) = ctx.saved_tensors  # type: ignore[attr-defined]
        return grad_output * 2 * x


def square(x: Tensor) -> Tensor:
    """x floating [...] -> [...]，使用自定义 autograd Function。"""
    return Square.apply(x)


if __name__ == "__main__":
    x = torch.tensor([2.0], requires_grad=True)
    square(x).sum().backward()
    print(x.grad)  # [1]，值为 4
