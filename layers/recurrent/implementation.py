"""RNN/LSTM/GRU 单步计算，gate 顺序与 PyTorch 一致。"""

import torch
from torch import Tensor, nn


class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """x [B,I], h [B,H] -> h_new [B,H]。"""
        return torch.tanh(self.input(x) + self.hidden(h))


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_size, 4 * hidden_size)
        self.hidden = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """x [B,I], state=(h,c)，各 [B,H] -> (h_new,c_new)，各 [B,H]。"""
        h, c = state
        i, f, g, o = (self.input(x) + self.hidden(h)).chunk(4, dim=-1)  # 各 [B,H]
        next_c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        next_h = o.sigmoid() * next_c.tanh()
        return next_h, next_c


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_size, 3 * hidden_size)
        self.hidden = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """x [B,I], h [B,H] -> h_new [B,H]；reset gate 乘在 hidden 投影之后。"""
        xr, xz, xn = self.input(x).chunk(3, dim=-1)  # 各 [B,H]
        hr, hz, hn = self.hidden(h).chunk(3, dim=-1)
        r, z = (xr + hr).sigmoid(), (xz + hz).sigmoid()
        n = (xn + r * hn).tanh()
        return (1 - z) * n + z * h


if __name__ == "__main__":
    print(GRUCell(3, 4)(torch.randn(2, 3), torch.zeros(2, 4)).shape)
