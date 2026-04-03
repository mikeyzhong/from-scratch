import math

import torch
import torch.nn as nn


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
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q, k, v: (batch, seq_len, dim)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (batch, seq_len, seq_len)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v  # (batch, seq_len, dim)
