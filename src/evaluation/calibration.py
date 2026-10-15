from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = predictions == y_true
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if not np.any(mask):
            continue
        ece += np.mean(mask) * abs(np.mean(accuracies[mask]) - np.mean(confidences[mask]))
    return float(ece)


def brier_score_multiclass(y_true, probabilities) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
