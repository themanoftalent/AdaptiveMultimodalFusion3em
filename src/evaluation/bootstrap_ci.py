from __future__ import annotations

import numpy as np


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap: int = 1000, seed: int = 42, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lower, upper
