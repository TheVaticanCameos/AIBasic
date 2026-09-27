"""稳定的指数归约。所有操作保留 autograd 计算图。"""

import torch
from torch import Tensor


def logsumexp(x: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """x: floating [..., D, ...]；输出沿 dim 归约，要求输入为有限值。"""
    maximum = x.amax(dim=dim, keepdim=True)  # [..., 1, ...]
    result = maximum + (x - maximum).exp().sum(dim=dim, keepdim=True).log()
    return result if keepdim else result.squeeze(dim)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """x、返回值: 同形状 floating tensor；dim 是类别维度。"""
    # 不能先把巨大 maximum 加回再相减，否则 float32 会丢失 log(sum(exp))。
    shifted = x - x.amax(dim=dim, keepdim=True)  # 与 x 同形状
    return shifted - shifted.exp().sum(dim=dim, keepdim=True).log()


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """x、返回值: 同形状；沿 dim 的概率和为 1。"""
    return log_softmax(x, dim).exp()


if __name__ == "__main__":
    print(softmax(torch.tensor([[1000.0, 1001.0]])))
