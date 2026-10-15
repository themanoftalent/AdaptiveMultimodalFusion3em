from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd
import torch
import numpy as np

from src.evaluation.metrics import compute_classification_metrics, probabilities_to_predictions
from src.fusion.adaptive_gate import adaptive_fusion
from src.models.gating_network import AdaptiveGatingNetwork
from src.utils.io import load_yaml, save_json, set_seed, resolve_device, ensure_dir

PROB_COLS = ["prob_non_tumor", "prob_viable_tumor", "prob_necrotic_tumor"]


def read_probs(path):
    frame = pd.read_csv(path)
    return frame[PROB_COLS].to_numpy(dtype="float32"), frame["label"].to_numpy(dtype="int64"), frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adaptive gate on P004.")
    parser.add_argument("--config", default="configs/fusion_gate.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("project", {}).get("seed", 42))
    device = resolve_device(cfg.get("project", {}).get("device", "auto"))
    paths = cfg["paths"]
    p_h, y, frame_h = read_probs(paths["histopathology_probabilities_p004"])
    p_r, _, _ = read_probs(paths["radiograph_probabilities_p004"])
    n = min(len(p_h), len(p_r), len(y))
    p_h_t = torch.tensor(p_h[:n], device=device)
    p_r_t = torch.tensor(p_r[:n], device=device)
    y = y[:n]

    gate = AdaptiveGatingNetwork(hidden_dims=cfg["adaptive_gate"].get("hidden_dims", [128, 64])).to(device)
    state = torch.load(paths["gate_checkpoint"], map_location=device)
    gate.load_state_dict(state["model_state_dict"])
    gate.eval()
    with torch.no_grad():
        alpha = gate(p_r_t, p_h_t)
        fused = adaptive_fusion(p_r_t, p_h_t, alpha).cpu().numpy()
        alpha_np = alpha.cpu().numpy().reshape(-1)
    pred = probabilities_to_predictions(fused)
    
    # Compute fixed-alpha fusion baseline for McNemar test comparison
    from src.fusion.fixed_alpha import fixed_alpha_fusion
    fixed_alpha_val = cfg["fusion"].get("fixed_alpha", 0.25)
    fused_fixed = fixed_alpha_fusion(p_r[:n], p_h[:n], alpha=fixed_alpha_val)
    pred_fixed = probabilities_to_predictions(fused_fixed)

    # Compute McNemar's continuity corrected test
    from src.evaluation.mcnemar import mcnemar_continuity_corrected
    mcnemar_result = mcnemar_continuity_corrected(y, pred_fixed, pred)

    metrics = compute_classification_metrics(y, pred, fused)
    metrics["n"] = int(n)
    metrics["correct"] = int(np.sum(pred == y))
    metrics["mean_alpha"] = float(np.mean(alpha_np))
    metrics["sd_alpha"] = float(np.std(alpha_np, ddof=1)) if n > 1 else 0.0
    metrics["mcnemar"] = mcnemar_result

    out_dir = ensure_dir(paths.get("metrics_dir", "results/metrics"))
    save_json(metrics, Path(out_dir) / "adaptive_gate_p004_metrics.json")
    out_frame = frame_h.iloc[:n].copy()
    out_frame["predicted_label"] = pred
    out_frame["alpha_hat"] = alpha_np
    out_frame.to_csv("results/probabilities/adaptive_gate_p004_predictions.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
