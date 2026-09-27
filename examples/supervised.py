"""CPU 合成数据：MLP 分类、CNN 图案分类、微型 Transformer next-token 训练。"""

import argparse
import json
from typing import Literal

import torch
from torch import Tensor, nn

from attention.positional.implementation import sinusoidal_encoding
from attention.scaled_dot_product.implementation import causal_mask
from attention.transformer.implementation import TransformerBlock
from layers.embedding.implementation import Embedding
from layers.linear.implementation import Linear
from layers.residual.implementation import ResidualBlock
from losses.classification.implementation import cross_entropy
from optimizers.adam.implementation import Adam
from training.loop.implementation import train_step


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = ResidualBlock(1, 8)
        self.head = Linear(8 * 8 * 8, 2)

    def forward(self, x: Tensor) -> Tensor:
        """x [B,1,8,8] -> logits [B,2]。"""
        return self.head(self.block(x).flatten(1))


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 8, dim: int = 16) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, dim)
        self.block = TransformerBlock(dim, 2, dim * 2)
        self.head = Linear(dim, vocab_size)
        self.dim = dim

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens int64 [B,T] -> logits [B,T,V]，仅关注当前位置与过去。"""
        _, t = tokens.shape
        x = self.embedding(tokens) + sinusoidal_encoding(t, self.dim, device=tokens.device)
        x = self.block(x, causal_mask(t, t, device=tokens.device))
        return self.head(x)


def train(
    kind: Literal["mlp", "cnn", "transformer"] = "mlp", steps: int = 100, seed: int = 0
) -> dict[str, float]:
    """固定小训练集，用于验证模块组合能学习；不代表泛化评测。"""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if kind == "mlp":
        model: nn.Module = nn.Sequential(Linear(4, 16), nn.ReLU(), Linear(16, 2))
        x = torch.randn(64, 4)  # [B,I]
        y = (x[:, 0] + x[:, 1] > 0).long()  # [B]
    elif kind == "cnn":
        model = TinyCNN()
        y = torch.arange(32) % 2  # [B]
        x = torch.randn(32, 1, 8, 8) * 0.1
        x[y == 0, :, :, 3:5] += 1  # 垂直线
        x[y == 1, :, 3:5, :] += 1  # 水平线
    else:
        model = TinyLanguageModel()
        starts = torch.randint(8, (32, 1))
        sequence = (starts + torch.arange(9)[None, :]) % 8  # [B,T+1]
        x, y = sequence[:, :-1], sequence[:, 1:]  # 各 [B,T]，目标右移一位

    def loss_fn(logits: Tensor, labels: Tensor) -> Tensor:
        """logits [B,...,C], labels [B,...] -> []；展平所有分类位置。"""
        return cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

    optimizer = Adam(model.parameters(), lr=0.01)
    model.eval()
    with torch.no_grad():
        initial = float(loss_fn(model(x), y))
    for _ in range(steps):
        train_step(model, [(x, y)], optimizer, loss_fn, max_grad_norm=1.0)
    model.eval()
    with torch.no_grad():
        output = model(x)
        final = float(loss_fn(output, y))
        accuracy = float((output.argmax(-1) == y).float().mean())
    return {"initial_loss": initial, "final_loss": final, "training_accuracy": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mlp", "cnn", "transformer"], default="mlp")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train(args.model, args.steps, args.seed), indent=2))
