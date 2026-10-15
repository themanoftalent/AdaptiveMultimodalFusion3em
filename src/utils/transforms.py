from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms


def minmax_tensor(tensor: torch.Tensor) -> torch.Tensor:
    low = tensor.amin(dim=(-2, -1), keepdim=True)
    high = tensor.amax(dim=(-2, -1), keepdim=True)
    return (tensor - low) / (high - low + 1e-8)


def build_transforms(input_size: int = 224, modality: str = "histopathology", train: bool = False):
    ops = [transforms.Resize((input_size, input_size))]

    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0)),
        ])
    else:
        ops.append(transforms.CenterCrop(input_size))

    ops.append(transforms.ToTensor())

    if modality == "radiograph":
        ops.append(transforms.Lambda(minmax_tensor))
    else:
        # Macenko normalization is implemented as an offline preprocessing option.
        # ImageNet normalization is used here for EfficientNet-B0 input scaling.
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(ops)
