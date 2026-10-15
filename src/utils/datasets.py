from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageCSVDataset(Dataset):
    """Image dataset backed by a CSV file.

    Required CSV columns:
        image_path, label

    Optional CSV columns:
        image_id, label_name, patient_id, split, source
    """

    def __init__(self, csv_path: str | Path, transform: Callable | None = None):
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(self.csv_path)
        if "image_path" not in self.frame.columns or "label" not in self.frame.columns:
            raise ValueError(f"{csv_path} must contain image_path and label columns.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        image_id = row.get("image_id", str(index))
        return image, label, image_id
