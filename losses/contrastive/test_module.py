import torch

from losses.contrastive.implementation import info_nce


def test_matched_pairs_better_and_symmetric() -> None:
    x = torch.eye(5, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    assert info_nce(x, y) < info_nce(x, y.roll(1, 0))
    torch.testing.assert_close(info_nce(x, y), info_nce(y, x))
    assert torch.autograd.gradcheck(info_nce, (x, y))
