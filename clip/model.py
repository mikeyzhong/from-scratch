import torch.nn as nn

from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from tokenizer import CLIPTokenizer
from loss import CLIPLoss


class CLIP(nn.Module):
    def __init__(self, projection_dim: int = 512):
        super().__init__()
        self.tokenizer = CLIPTokenizer()
        self.image_encoder = ImageEncoder(projection_dim=projection_dim)
        self.text_encoder = TextEncoder(vocab_size=self.tokenizer.vocab_size, projection_dim=projection_dim)
        self.loss_fn = CLIPLoss()

    def forward(self, images, captions):
        tokens = self.tokenizer(captions).to(images.device)

        image_embeds = self.image_encoder(images)
        text_embeds = self.text_encoder(tokens)

        loss = self.loss_fn(image_embeds, text_embeds)
        return loss
