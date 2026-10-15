from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def mcnemar_continuity_corrected(y_true, pred_a, pred_b) -> dict[str, float | int]:
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    statistic = 0.0 if b + c == 0 else (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value = float(1.0 - chi2.cdf(statistic, df=1))
    return {"b": b, "c": c, "chi2": float(statistic), "p_value": p_value}
