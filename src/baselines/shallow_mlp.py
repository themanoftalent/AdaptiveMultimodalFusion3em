from __future__ import annotations

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_shallow_mlp(hidden_layer_sizes=(64,), max_iter: int = 500, random_state: int = 42):
    """Builds a standardized Shallow Multilayer Perceptron baseline pipeline."""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
    )
