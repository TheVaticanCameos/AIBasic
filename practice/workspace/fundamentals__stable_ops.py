"""稳定 softmax / log-softmax / logsumexp：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor

def logsumexp(x: Tensor, dim: int=-1, keepdim: bool=False) -> Tensor:
    """x: floating [..., D, ...]；输出沿 dim 归约，要求输入为有限值。"""
    # raise NotImplementedError('手写练习：请完成核心实现')
    maximum = x.amax(dim=dim, keepdim=True)
    logsumexp = maximum + (x - maximum).exp().sum(dim=dim, keepdim=True).log()
    return logsumexp if keepdim else logsumexp.squeeze(dim)

def log_softmax(x: Tensor, dim: int=-1) -> Tensor:
    """x、返回值: 同形状 floating tensor；dim 是类别维度。"""
    # raise NotImplementedError('手写练习：请完成核心实现')
    shifted = x - x.amax(dim=dim, keepdim=True)
    return shifted - shifted.exp().sum(dim=dim, keepdim=True).log()

def softmax(x: Tensor, dim: int=-1) -> Tensor:
    """x、返回值: 同形状；沿 dim 的概率和为 1。"""
    # raise NotImplementedError('手写练习：请完成核心实现')
    return log_softmax(x, dim).exp()


if __name__ == "__main__":
    print(softmax(torch.tensor([[1000.0, 1001.0]])))
