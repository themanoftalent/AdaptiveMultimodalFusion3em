from __future__ import annotations

import numpy as np


def confidence_attention_late_fusion(p_r: np.ndarray, p_h: np.ndarray) -> np.ndarray:
    """Attention-style late fusion using normalized branch confidence scores.

    This is a lightweight reproducibility baseline. It does not use raw images.
    """
    c_r = p_r.max(axis=1, keepdims=True)
    c_h = p_h.max(axis=1, keepdims=True)
    denom = c_r + c_h + 1e-8
    w_r = c_r / denom
    w_h = c_h / denom
    return w_r * p_r + w_h * p_h
