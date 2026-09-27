"""教学 SGD，支持 Momentum；不依赖 torch.optim 完成更新。"""

from collections.abc import Iterable

import torch
from torch import Tensor


class SGD:
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0 or not 0 <= momentum < 1 or weight_decay < 0:
            raise ValueError("invalid optimizer hyperparameters")
        self.parameters = list(parameters)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.velocity: dict[int, Tensor] = {}  # 每个参数对应同形状的动量

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            gradient = parameter.grad + self.weight_decay * parameter  # 参数同形状 [...]
            if self.momentum:
                if index not in self.velocity:
                    self.velocity[index] = gradient.clone()
                else:
                    self.velocity[index].mul_(self.momentum).add_(gradient)
                gradient = self.velocity[index]
            parameter.add_(gradient, alpha=-self.lr)


if __name__ == "__main__":
    parameter = torch.tensor([2.0], requires_grad=True)
    optimizer = SGD([parameter], lr=0.1)
    parameter.square().sum().backward()
    optimizer.step()
    print(parameter)
