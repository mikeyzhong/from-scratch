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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Pair up features as (x_even, x_odd) and rotate them to (-x_odd, x_even).
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


# Step 2: Rotary position embedding (RoPE)
# Rotate q and k feature pairs by position-dependent angles before computing
# attention scores.
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires an even embedding dimension.")

        position = torch.arange(max_seq_len).float().unsqueeze(1)  # (max_seq_len, 1)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
        freqs = position * inv_freq  # (max_seq_len, dim/2)

        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)

        self.register_buffer("cos", cos, persistent=False)  # (max_seq_len, dim)
        self.register_buffer("sin", sin, persistent=False)  # (max_seq_len, dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (..., seq_len, dim)
        seq_len = q.size(-2)
        dim = q.size(-1)

        cos = self.cos[:seq_len].view(*([1] * (q.ndim - 2)), seq_len, dim)
        sin = self.sin[:seq_len].view(*([1] * (q.ndim - 2)), seq_len, dim)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


# Step 3: Scaled dot-product attention
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


# Step 4: Single-head attention with optional RoPE
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


# Step 5: Multi-head attention
# Split the model dimension across multiple heads, run attention in parallel,
# then merge the head outputs back together with a final projection.
class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 2048, use_rope: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = QKVProjection(dim)
        self.rope = RotaryPositionEmbedding(max_seq_len=max_seq_len, dim=self.head_dim) if use_rope else None
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.dim)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        q, k, v = self.qkv(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.rope is not None:
            q, k = self.rope(q, k)

        out = scaled_dot_product_attention(q, k, v, causal=causal)
        out = self._merge_heads(out)
        return self.out_proj(out)


def toy_multihead_attention_example() -> None:
    x = torch.tensor(
        [
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
                [1.0, 1.0, 0.5, 0.5],
            ]
        ]
    )  # (batch=1, seq=3, dim=4)

    attn = MultiHeadAttention(dim=4, num_heads=2, max_seq_len=8, use_rope=True)
    y = attn(x, causal=True)

    print("input shape:", x.shape)
    print("num_heads:", attn.num_heads)
    print("head_dim:", attn.head_dim)
    print("q split shape:", attn._split_heads(attn.qkv(x)[0]).shape)
    print("output shape:", y.shape)
    print("output:", y)


if __name__ == "__main__":
    toy_multihead_attention_example()
