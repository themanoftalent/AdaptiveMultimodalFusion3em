from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd
import numpy as np

from src.evaluation.calibration import expected_calibration_error, brier_score_multiclass
from src.evaluation.metrics import compute_classification_metrics, probabilities_to_predictions
from src.utils.io import save_json, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Restricted Hitit external validation interface.")
    parser.add_argument("--metadata", required=True, help="Authorized local metadata CSV. Do not commit private metadata.")
    parser.add_argument("--predictions", required=True, help="Authorized prediction CSV generated locally.")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    predictions = pd.read_csv(args.predictions)
    if "binary_label" not in metadata.columns:
        raise ValueError("Metadata must contain binary_label column.")
    required = ["prob_non_tumor", "prob_tumor"]
    if not set(required).issubset(predictions.columns):
        raise ValueError(f"Predictions must contain {required}")

    y = metadata["binary_label"].to_numpy(dtype=int)
    probs = predictions[required].to_numpy(dtype=float)
    pred = probabilities_to_predictions(probs)
    metrics = compute_classification_metrics(y, pred, probs)
    metrics["ece"] = expected_calibration_error(y, probs)
    metrics["brier"] = brier_score_multiclass(y, probs)
    save_json(metrics, ensure_dir("results/metrics") / "hitit_authorized_external_metrics.json")
    print(metrics)
    print("Reminder: Hitit data and authorized prediction files must not be uploaded to GitHub.")


if __name__ == "__main__":
    main()
