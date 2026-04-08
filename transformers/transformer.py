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


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        depth: int,
        max_seq_len: int = 2048,
        use_rope: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    max_seq_len=max_seq_len,
                    use_rope=use_rope,
                )
                for _ in range(depth)
            ]
        )
        self.ln_f = LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = self.token_embed(tokens)

        for block in self.blocks:
            x = block(x, causal=causal)

        x = self.ln_f(x)
        return self.lm_head(x)


def main() -> None:
    model = Transformer(vocab_size=32, dim=8, num_heads=2, depth=2, max_seq_len=16, use_rope=True)
    tokens = torch.randint(0, model.vocab_size, (2, 5))
    logits = model(tokens, causal=True)

    assert logits.shape == (2, 5, model.vocab_size)

    print("transformer scaffold ready")
    print("token input shape:", tokens.shape)
    print("logits shape:", logits.shape)


if __name__ == "__main__":
    main()
