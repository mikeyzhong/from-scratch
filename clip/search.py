from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from model import CLIP
from dataset import CLIPDataset


def search(query: str, top_k: int = 5):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = CLIP().to(device)
    model.load_state_dict(torch.load("clip_model.pt", map_location=device))
    model.eval()

    dataset = CLIPDataset(Path("data/captions.txt"), Path("data/Images"), train=False)

    # encode the text query
    tokens = model.tokenizer(query).to(device)
    with torch.no_grad():
        text_embed = model.text_encoder(tokens)
        text_embed = F.normalize(text_embed, dim=-1)

    # encode all images
    all_image_embeds = []
    all_filenames = []
    with torch.no_grad():
        for i in range(0, len(dataset), 32):
            batch_images = []
            for j in range(i, min(i + 32, len(dataset))):
                img, _ = dataset[j]
                batch_images.append(img)
                if dataset.pairs[j][0] not in all_filenames:
                    all_filenames.append(dataset.pairs[j][0])

            images = torch.stack(batch_images).to(device)
            embeds = model.image_encoder(images)
            embeds = F.normalize(embeds, dim=-1)
            all_image_embeds.append(embeds.cpu())

    all_image_embeds = torch.cat(all_image_embeds)

    # deduplicate to unique images
    seen = {}
    for i, fname in enumerate(all_filenames):
        if fname not in seen:
            seen[fname] = i
    unique_indices = list(seen.values())
    unique_filenames = list(seen.keys())
    unique_embeds = all_image_embeds[unique_indices]

    # compute similarities
    similarities = (text_embed.cpu() @ unique_embeds.t()).squeeze(0)
    top_indices = similarities.argsort(descending=True)[:top_k]

    print(f"\nQuery: '{query}'\n")
    for rank, idx in enumerate(top_indices):
        fname = unique_filenames[idx]
        score = similarities[idx].item()
        print(f"  {rank + 1}. {fname} (score: {score:.3f})")


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "a dog playing fetch"
    search(query)
