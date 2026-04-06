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


def main() -> None:
    x = torch.randn(2, 5, 8)
    attn = MultiHeadAttention(dim=8, num_heads=2, max_seq_len=16, use_rope=True)
    ffn = FeedForward(dim=8)

    y = attn(x, causal=True)
    z = ffn(y)

    print("transformer scaffold ready")
    print("example input shape:", x.shape)
    print("attention output shape:", y.shape)
    print("ffn output shape:", z.shape)


if __name__ == "__main__":
    main()
