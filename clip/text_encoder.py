import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, max_length=77, embed_dim=192, depth=4, num_heads=3, projection_dim=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_length, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, projection_dim)

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)

        x = self.token_embed(tokens) + self.pos_embed(positions)

        mask = (tokens == 0)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)

        eot_idx = (~mask).sum(dim=1) - 1
        x = x[torch.arange(batch_size, device=x.device), eot_idx]

        x = self.proj(x)
        return x
