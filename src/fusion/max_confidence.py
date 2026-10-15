from __future__ import annotations

import numpy as np
import torch


def max_confidence_selection(p_r, p_h):
    """Selects the full probability vector from the more confident branch."""
    if isinstance(p_r, torch.Tensor):
        max_r = torch.max(p_r, dim=1, keepdim=True).values
        max_h = torch.max(p_h, dim=1, keepdim=True).values
        mask = (max_r > max_h).float()
        return mask * p_r + (1.0 - mask) * p_h
    max_r = np.max(p_r, axis=1, keepdims=True)
    max_h = np.max(p_h, axis=1, keepdims=True)
    mask = (max_r > max_h).astype(float)
    return mask * p_r + (1.0 - mask) * p_h
