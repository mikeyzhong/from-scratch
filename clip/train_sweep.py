from pathlib import Path

import torch
import torch.nn.functional as F
import wandb

from model import CLIP
from dataset import get_dataloader


def measure_collapse(model, dataset, device, num_samples=100):
    """Measure image embedding collapse. Returns mean pairwise similarity (0=diverse, 1=collapsed)."""
    model.eval()
    images = torch.stack([dataset[i][0] for i in range(min(num_samples, len(dataset)))]).to(device)
    with torch.no_grad():
        embeds = model.image_encoder(images)
        embeds = F.normalize(embeds, dim=-1)
        sims = embeds @ embeds.t()
        # exclude diagonal (self-similarity = 1)
        mask = ~torch.eye(sims.shape[0], dtype=torch.bool, device=device)
        mean_sim = sims[mask].mean().item()
    model.train()
    return mean_sim


def train():
    defaults = {
        "batch_size": 32,
        "lr": 1e-4,
        "warmup_epochs": 0,
        "grad_clip": 0.0,
    }
    wandb.init(config=defaults)
    config = wandb.config
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("epoch")
    wandb.define_metric("epoch/*", step_metric="epoch")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = CLIP().to(device)
    dataloader = get_dataloader(
        Path("data/captions.txt"),
        Path("data/Images"),
        batch_size=config.batch_size,
        num_workers=4,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

    # warmup scheduler
    num_epochs = 10
    steps_per_epoch = len(dataloader)
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for images, captions in dataloader:
            images = images.to(device)

            loss = model(images, captions)

            optimizer.zero_grad()
            loss.backward()

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            step += 1

            wandb.log({
                "train/step": step,
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch + 1,
            })

        avg_loss = total_loss / num_batches
        collapse = measure_collapse(model, dataloader.dataset, device)

        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "image_collapse": collapse,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch/loss": avg_loss,
            "epoch/image_collapse": collapse,
            "epoch/lr": optimizer.param_groups[0]["lr"],
        })

        print(f"Epoch {epoch + 1}/{num_epochs} — loss: {avg_loss:.4f}, collapse: {collapse:.4f}")

    wandb.finish()


if __name__ == "__main__":
    train()
