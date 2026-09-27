import pytest
import torch
import torch.nn.functional as F

from layers.conv2d.implementation import conv2d_loop, conv2d_unfold


@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1)])
def test_loop_unfold_torch_outputs_and_all_gradients(stride: int, padding: int) -> None:
    inputs = (
        torch.randn(2, 2, 5, 6, dtype=torch.double),
        torch.randn(3, 2, 3, 3, dtype=torch.double),
        torch.randn(3, dtype=torch.double),
    )
    reference_inputs = tuple(x.clone().requires_grad_() for x in inputs)
    expected = F.conv2d(*reference_inputs, stride=stride, padding=padding)
    probe = torch.randn_like(expected)
    expected_grads = torch.autograd.grad((expected * probe).sum(), reference_inputs)
    for implementation in (conv2d_loop, conv2d_unfold):
        actual_inputs = tuple(x.clone().requires_grad_() for x in inputs)
        actual = implementation(*actual_inputs, stride=stride, padding=padding)
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad((actual * probe).sum(), actual_inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)
