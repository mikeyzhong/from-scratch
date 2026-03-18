import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=192, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.proj(x)       # (B, 192, 8, 8)
        x = x.flatten(2)       # (B, 192, 64)
        x = x.transpose(1, 2)  # (B, 64, 192)
        return x
