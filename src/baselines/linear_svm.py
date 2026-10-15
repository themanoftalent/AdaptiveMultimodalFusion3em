from __future__ import annotations

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_linear_svm(C: float = 1.0, random_state: int = 42):
    """Builds a standardized Linear Support Vector Classifier baseline pipeline."""
    return make_pipeline(
        StandardScaler(),
        LinearSVC(C=C, random_state=random_state, dual="auto")
    )
