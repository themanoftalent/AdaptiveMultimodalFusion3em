from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_regression(C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
    """Builds a standardized Logistic Regression classifier baseline pipeline."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    )
