import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from attention.attention import MultiHeadAttention


def main() -> None:
    x = torch.randn(2, 5, 8)
    attn = MultiHeadAttention(dim=8, num_heads=2, max_seq_len=16, use_rope=True)
    y = attn(x, causal=True)

    print("transformer scaffold ready")
    print("example input shape:", x.shape)
    print("attention output shape:", y.shape)


if __name__ == "__main__":
    main()
