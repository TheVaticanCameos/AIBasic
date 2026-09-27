import torch

from layers.dropout.implementation import Dropout


def test_scaling_eval_and_edge_probabilities() -> None:
    layer = Dropout(0.25)
    x = torch.ones(100000, requires_grad=True)
    y = layer(x)
    assert abs(float(y.mean().detach()) - 1) < 0.01
    assert set(y.detach().unique().tolist()) == {0.0, float(torch.tensor(1 / 0.75))}
    y.sum().backward()
    torch.testing.assert_close(x.grad, y.detach())
    torch.testing.assert_close(layer.eval()(x), x)
    torch.testing.assert_close(Dropout(0)(x), x)
    torch.testing.assert_close(Dropout(1)(x), torch.zeros_like(x))
