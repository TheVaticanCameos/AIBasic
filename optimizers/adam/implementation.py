"""Adam / AdamW：通过 decoupled_weight_decay 区分 L2 与解耦衰减。"""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AdamState:
    step: int
    first: Tensor  # 与参数同形状 [...]
    second: Tensor  # 与参数同形状 [...]


class Adam:
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
    ) -> None:
        if lr < 0 or eps <= 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("invalid optimizer hyperparameters")
        self.parameters = list(parameters)
        self.lr, self.betas, self.eps = lr, betas, eps
        self.weight_decay = weight_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.state: dict[int, AdamState] = {}

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        b1, b2 = self.betas
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            gradient = parameter.grad  # [...]
            if self.decoupled_weight_decay:
                parameter.mul_(1 - self.lr * self.weight_decay)
            else:
                gradient = gradient + self.weight_decay * parameter
            if index not in self.state:
                self.state[index] = AdamState(
                    0, torch.zeros_like(parameter), torch.zeros_like(parameter)
                )
            state = self.state[index]
            state.step += 1  # 每个参数独立计数，grad=None 时不递增
            state.first.mul_(b1).add_(gradient, alpha=1 - b1)
            state.second.mul_(b2).addcmul_(gradient, gradient, value=1 - b2)
            first_hat = state.first / (1 - b1**state.step)
            second_hat = state.second / (1 - b2**state.step)
            parameter.addcdiv_(first_hat, second_hat.sqrt() + self.eps, value=-self.lr)


class AdamW(Adam):
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(parameters, lr, betas, eps, weight_decay, True)


if __name__ == "__main__":
    parameter = torch.tensor([2.0], requires_grad=True)
    optimizer = AdamW([parameter])
    parameter.square().sum().backward()
    optimizer.step()
    print(parameter)
