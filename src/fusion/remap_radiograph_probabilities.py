from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class RadiographRemapper(nn.Module):
    """Converts binary radiograph logits into three-class probabilities."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return remap_binary_probabilities_tensor(probs)


def remap_binary_probabilities_tensor(probs: torch.Tensor) -> torch.Tensor:
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError("Binary radiograph probabilities must have shape [N, 2].")
    p_other = probs[:, 0:1]
    p_tumor = probs[:, 1:2]
    return torch.cat([p_other, p_tumor / 2.0, p_tumor / 2.0], dim=1)


def remap_binary_probabilities_array(probs: np.ndarray) -> np.ndarray:
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError("Binary radiograph probabilities must have shape [N, 2].")
    p_other = probs[:, [0]]
    p_tumor = probs[:, [1]]
    return np.concatenate([p_other, p_tumor / 2.0, p_tumor / 2.0], axis=1)


def remap_probability_csv(input_csv: str, output_csv: str) -> pd.DataFrame:
    frame = pd.read_csv(input_csv)
    prob_cols = ["prob_other", "prob_tumor"]
    if not set(prob_cols).issubset(frame.columns):
        raise ValueError(f"Input CSV must contain columns: {prob_cols}")
    remapped = remap_binary_probabilities_array(frame[prob_cols].to_numpy(dtype=float))
    out = frame.copy()
    out["prob_non_tumor"] = remapped[:, 0]
    out["prob_viable_tumor"] = remapped[:, 1]
    out["prob_necrotic_tumor"] = remapped[:, 2]
    out.to_csv(output_csv, index=False)
    return out
