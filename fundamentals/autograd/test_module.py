import torch

from fundamentals.autograd.implementation import square


def test_backward_and_double_backward() -> None:
    x = torch.randn(3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(square, (x,))
    assert torch.autograd.gradgradcheck(square, (x,))


def test_accumulation_detach_and_no_grad() -> None:
    x = torch.tensor([2.0], requires_grad=True)
    square(x).sum().backward()
    square(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([8.0]))
    assert not x.detach().requires_grad
    with torch.no_grad():
        assert not (x * 2).requires_grad
