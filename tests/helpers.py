"""测试共用的数值/梯度比较；不被实现模块依赖。"""

from collections.abc import Callable

import torch
from torch import Tensor


def compare_input_gradients(
    actual_fn: Callable[[Tensor], Tensor], expected_fn: Callable[[Tensor], Tensor], x: Tensor
) -> None:
    actual_x = x.detach().clone().requires_grad_(True)
    expected_x = x.detach().clone().requires_grad_(True)
    actual, expected = actual_fn(actual_x), expected_fn(expected_x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(actual_x.grad, expected_x.grad, rtol=1e-6, atol=1e-8)
