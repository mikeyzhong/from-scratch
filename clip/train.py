from pathlib import Path

import torch

from model import CLIP
from dataset import get_dataloader


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CLIP().to(device)
    dataloader = get_dataloader(
        Path("data/captions.txt"),
        Path("data/Images"),
        batch_size=32,
        num_workers=4,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for images, captions in dataloader:
            images = images.to(device)

            loss = model(images, captions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs} — loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "clip_model.pt")
    print("Saved model to clip_model.pt")


if __name__ == "__main__":
    train()
