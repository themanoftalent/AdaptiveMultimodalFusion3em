from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier


def build_xgboost_probability_baseline(**kwargs):
    params = {
        "max_depth": 2,
        "n_estimators": 100,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 0.5,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42,
    }
    params.update(kwargs)
    return XGBClassifier(**params)


def concatenate_probability_vectors(p_r: np.ndarray, p_h: np.ndarray) -> np.ndarray:
    return np.concatenate([p_r, p_h], axis=1)
