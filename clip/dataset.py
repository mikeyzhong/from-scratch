from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CLIPDataset(Dataset):
    def __init__(self, captions_file: Path, images_dir: Path, image_size: int = 224, train: bool = True):
        df = pd.read_csv(captions_file)
        df["caption"] = df["caption"].str.strip().str.rstrip(".")

        # drop rows where image file is missing
        df = df[df["image"].apply(lambda f: (images_dir / f).exists())]
        self.pairs = list(zip(df["image"], df["caption"]))
        self.images_dir = images_dir

        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        filename, caption = self.pairs[idx]
        image = Image.open(self.images_dir / filename).convert("RGB")
        image = self.transform(image)
        return image, caption


def get_dataloader(
    captions_file: Path,
    images_dir: Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    image_size: int = 224,
    train: bool = True,
) -> DataLoader:
    dataset = CLIPDataset(captions_file, images_dir, image_size=image_size, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
