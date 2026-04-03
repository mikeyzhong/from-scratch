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
