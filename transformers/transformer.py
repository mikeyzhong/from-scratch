import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from attention.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or (4 * dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.down = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 2048, use_rope: bool = True):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
        )
        self.ln2 = LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal=causal)
        x = x + self.ffn(self.ln2(x))
        return x


def main() -> None:
    x = torch.randn(2, 5, 8)
    block = TransformerBlock(dim=8, num_heads=2, max_seq_len=16, use_rope=True)
    y = block(x, causal=True)

    print("transformer scaffold ready")
    print("example input shape:", x.shape)
    print("block output shape:", y.shape)


if __name__ == "__main__":
    main()
