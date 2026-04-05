import math

import torch
import torch.nn as nn


# Step 1: Learned absolute position embedding (GPT-2, BERT)
# Each position gets its own trainable vector, added to the token embedding.
class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions)


# Step 2: Sinusoidal position embedding (original Transformer, Vaswani 2017)
# Fixed sin/cos waves at different frequencies — no learned parameters.
# PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(max_seq_len).unsqueeze(1).float()  # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))  # (dim/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims: cos

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        return x + self.pe[:, :x.size(1)]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Pair up features as (x_even, x_odd) and rotate them to (-x_odd, x_even).
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


# Step 3: Rotary position embedding (RoPE)
# Instead of adding a position vector, RoPE rotates q and k feature pairs by
# position-dependent angles before the attention score is computed.
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
