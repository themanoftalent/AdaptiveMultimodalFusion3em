from __future__ import annotations

import numpy as np
import torch


def fixed_alpha_fusion(p_r, p_h, alpha: float = 0.25):
    """Computes fixed-alpha decision-level fusion."""
    return alpha * p_r + (1.0 - alpha) * p_h


def grid_search_fixed_alpha(p_r: np.ndarray, p_h: np.ndarray, labels: np.ndarray, step: float = 0.05):
    candidates = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 10)
    best_alpha = 0.0
    best_accuracy = -1.0
    for alpha in candidates:
        fused = fixed_alpha_fusion(p_r, p_h, float(alpha))
        preds = np.argmax(fused, axis=1)
        acc = float(np.mean(preds == labels))
        if acc > best_accuracy:
            best_accuracy = acc
            best_alpha = float(alpha)
    return best_alpha, best_accuracy
