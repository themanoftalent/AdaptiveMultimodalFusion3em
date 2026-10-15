from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import json
import pandas as pd
import numpy as np

from src.baselines.xgboost_probability_baseline import build_xgboost_probability_baseline, concatenate_probability_vectors
from src.baselines.logistic_regression import build_logistic_regression
from src.baselines.linear_svm import build_linear_svm
from src.baselines.shallow_mlp import build_shallow_mlp
from src.baselines.attention_late_fusion import confidence_attention_late_fusion
from src.evaluation.metrics import compute_classification_metrics, probabilities_to_predictions
from src.fusion.fixed_alpha import fixed_alpha_fusion
from src.fusion.max_confidence import max_confidence_selection
from src.utils.io import load_yaml, ensure_dir, save_json, set_seed

PROB_COLS = ["prob_non_tumor", "prob_viable_tumor", "prob_necrotic_tumor"]


def read_probs(path):
    frame = pd.read_csv(path)
    return frame[PROB_COLS].to_numpy(dtype=float), frame["label"].to_numpy(dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run decision-level fusion baselines.")
    parser.add_argument("--config", default="configs/baselines.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("project", {}).get("seed", 42))
    fusion_cfg = load_yaml(cfg["paths"].get("fusion_config", "configs/fusion_gate.yaml"))
    paths = fusion_cfg["paths"]
    out_dir = ensure_dir(cfg["paths"].get("output_dir", "results/metrics"))

    p_h_train, y_train = read_probs(paths["histopathology_probabilities_train"])
    p_r_train, _ = read_probs(paths["radiograph_probabilities_train"])
    p_h_test, y_test = read_probs(paths["histopathology_probabilities_p004"])
    p_r_test, _ = read_probs(paths["radiograph_probabilities_p004"])
    n_train = min(len(p_h_train), len(p_r_train), len(y_train))
    n_test = min(len(p_h_test), len(p_r_test), len(y_test))
    p_h_train, p_r_train, y_train = p_h_train[:n_train], p_r_train[:n_train], y_train[:n_train]
    p_h_test, p_r_test, y_test = p_h_test[:n_test], p_r_test[:n_test], y_test[:n_test]

    results = {}
    for name, probs in {
        "fixed_alpha": fixed_alpha_fusion(p_r_test, p_h_test, alpha=fusion_cfg["fusion"].get("fixed_alpha", 0.25)),
        "max_confidence": max_confidence_selection(p_r_test, p_h_test),
        "attention_late_fusion": confidence_attention_late_fusion(p_r_test, p_h_test),
    }.items():
        y_pred = probabilities_to_predictions(probs)
        results[name] = compute_classification_metrics(y_test, y_pred, probs)

    X_train = concatenate_probability_vectors(p_r_train, p_h_train)
    X_test = concatenate_probability_vectors(p_r_test, p_h_test)

    model_builders = {
        "xgboost": build_xgboost_probability_baseline(**cfg.get("xgboost", {})),
        "logistic_regression": build_logistic_regression(**cfg.get("logistic_regression", {})),
        "linear_svm": build_linear_svm(**cfg.get("linear_svm", {})),
        "shallow_mlp": build_shallow_mlp(**cfg.get("shallow_mlp", {})),
    }
    for name, model in model_builders.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = compute_classification_metrics(y_test, y_pred)

    save_json(results, out_dir / "baseline_metrics_p004.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
