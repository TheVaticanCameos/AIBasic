"""Channel-first BatchNorm，支持 [B,C] / [B,C,L] / [B,C,H,W]。"""

import torch
from torch import Tensor, nn


class BatchNorm(nn.Module):
    running_mean: Tensor
    running_var: Tensor

    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))  # [C]
        self.bias = nn.Parameter(torch.zeros(channels))  # [C]
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x: Tensor) -> Tensor:
        """x [B,C,...] -> 同形状；每个 C 在 batch 与空间维度上归约。"""
        dims = (0, *range(2, x.ndim))
        shape = (1, x.shape[1], *([1] * (x.ndim - 2)))  # 广播到 [B,C,...]
        if self.training:
            count = x.numel() // x.shape[1]
            if count <= 1:
                raise ValueError("training requires more than one value per channel")
            mean = x.mean(dims)  # [C]
            var = x.var(dims, correction=0)  # [C]，前向用有偏估计
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                # PyTorch running_var 维护无偏估计。
                self.running_var.lerp_(var.detach() * count / (count - 1), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        normalized = (x - mean.reshape(shape)) * torch.rsqrt(var.reshape(shape) + self.eps)
        return normalized * self.weight.reshape(shape) + self.bias.reshape(shape)


if __name__ == "__main__":
    print(BatchNorm(3)(torch.randn(2, 3, 4, 4)).shape)
