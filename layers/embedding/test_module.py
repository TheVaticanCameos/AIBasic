import torch

from layers.embedding.implementation import Embedding


def test_repeated_indices_accumulate_gradients() -> None:
    embedding = Embedding(5, 3)
    output = embedding(torch.tensor([[1, 2, 1]]))
    torch.testing.assert_close(output[0, 0], embedding.weight[1])
    output.sum().backward()
    expected = torch.zeros(5, 3)
    expected[1], expected[2] = 2, 1
    torch.testing.assert_close(embedding.weight.grad, expected)
