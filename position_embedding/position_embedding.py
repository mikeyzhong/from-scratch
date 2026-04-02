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
