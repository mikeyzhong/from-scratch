import math

import torch
import torch.nn as nn

from position_embedding.position_embedding import RotaryPositionEmbedding


# Step 1: Q, K, V projections
# Each token vector gets projected into three separate vectors:
#   Q (query) — "what am I looking for?"
#   K (key)   — "what do I contain?"
#   V (value) — "what do I offer if selected?"
class QKVProjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, dim)
        return self.w_q(x), self.w_k(x), self.w_v(x)


# Step 2: Scaled dot-product attention
# For every token, compute how much it should attend to every other token,
# then return a weighted sum of their values.
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    # q, k, v: (batch, seq_len, dim)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (batch, seq_len, seq_len)

    if causal:
        seq_len = q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return weights @ v  # (batch, seq_len, dim)


# Step 3: Single-head attention with optional RoPE
class SingleHeadAttention(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, use_rope: bool = False):
        super().__init__()
        self.qkv = QKVProjection(dim)
        self.rope = RotaryPositionEmbedding(max_seq_len=max_seq_len, dim=dim) if use_rope else None

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        q, k, v = self.qkv(x)

        if self.rope is not None:
            q, k = self.rope(q, k)

        return scaled_dot_product_attention(q, k, v, causal=causal)
