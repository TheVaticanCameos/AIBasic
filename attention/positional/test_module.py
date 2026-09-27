import torch

from attention.positional.implementation import sinusoidal_encoding


def test_zero_position_and_offset_for_odd_dimensions() -> None:
    whole = sinusoidal_encoding(7, 5)
    torch.testing.assert_close(whole[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(whole[3:], sinusoidal_encoding(4, 5, offset=3))
