from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def probabilities_to_predictions(probabilities: np.ndarray) -> np.ndarray:
    return np.argmax(probabilities, axis=1)


def compute_classification_metrics(y_true, y_pred, probabilities=None) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if probabilities is not None:
        try:
            metrics["macro_auroc"] = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
        except Exception:
            metrics["macro_auroc"] = None
    return metrics
