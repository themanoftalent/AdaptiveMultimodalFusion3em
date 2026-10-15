from __future__ import annotations

import torch


def adaptive_fusion(p_r: torch.Tensor, p_h: torch.Tensor, alpha_hat: torch.Tensor) -> torch.Tensor:
    """Computes adaptive decision-level fusion."""
    return alpha_hat * p_r + (1.0 - alpha_hat) * p_h
