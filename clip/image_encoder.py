import torch.nn as nn
import timm


class ImageEncoder(nn.Module):
    def __init__(self, projection_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
        self.proj = nn.Linear(self.backbone.embed_dim, projection_dim)

    def forward(self, x):
        x = self.backbone(x)  # [batch, 192]
        x = self.proj(x)      # [batch, 512]
        return x
