import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from attention.attention import MultiHeadAttention


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
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
        )
        self.ln2 = nn.LayerNorm(dim)
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
