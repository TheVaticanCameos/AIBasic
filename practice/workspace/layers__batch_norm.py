"""BatchNorm：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn

class BatchNorm(nn.Module):
    running_mean: Tensor
    running_var: Tensor

    def __init__(self, channels: int, eps: float=1e-05, momentum: float=0.1) -> None:
        # raise NotImplementedError('手写练习：请完成核心实现')
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x: Tensor) -> Tensor:
        """x [B,C,...] -> 同形状；每个 C 在 batch 与空间维度上归约。"""
        # raise NotImplementedError('手写练习：请完成核心实现')
        b, c = x.shape[0], x.shape[1]
        dim = (0, *range(2, x.ndim))
        shape = (1, c, *([1] * (x.ndim - 2)))
        if self.training:
            count = x.numel() // c
            if count <= 1:
                raise ValueError("training requires more than 1 value per channel")
            mean = x.mean(dim)
            var = x.var(dim, unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_(var.detach() * count / (count - 1), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        normalized = (x - mean.reshape(shape)) / torch.sqrt(var.reshape(shape) + self.eps)
        return normalized * self.weight.reshape(shape) + self.bias.reshape(shape)
