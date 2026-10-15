from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.fusion.pseudo_optimal_alpha import compute_pseudo_optimal_alpha
from src.models.gating_network import AdaptiveGatingNetwork
from src.fusion.adaptive_gate import adaptive_fusion
from src.evaluation.metrics import compute_classification_metrics, probabilities_to_predictions
from src.utils.io import load_yaml, save_json, ensure_dir, set_seed, resolve_device

PROB_COLS = ["prob_non_tumor", "prob_viable_tumor", "prob_necrotic_tumor"]


def load_and_merge_patient_id(prob_path: str, tcia_index: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(prob_path)
    # Merge with index to retrieve patient_id mapping
    df = df.merge(tcia_index, on="image_id", how="left")
    # Fill missing patient IDs if any tile was not mapped
    df["patient_id"] = df["patient_id"].fillna("UNKNOWN")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Leave-One-Patient-Out (LOPO) cross-validation.")
    parser.add_argument("--config", default="configs/fusion_gate.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("project", {}).get("seed", 42))
    device = resolve_device(cfg.get("project", {}).get("device", "auto"))
    paths = cfg["paths"]

    # Verify all files are present
    tcia_index_path = Path("data/processed/tcia_index.csv")
    required_paths = [
        tcia_index_path,
        Path(paths["histopathology_probabilities_train"]),
        Path(paths["histopathology_probabilities_p003"]),
        Path(paths["histopathology_probabilities_p004"]),
        Path(paths["radiograph_probabilities_train"]),
        Path(paths["radiograph_probabilities_p003"]),
        Path(paths["radiograph_probabilities_p004"]),
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files for LOPO evaluation: {missing}. Run reproduction scripts 01-05 first.")

    tcia_index = pd.read_csv(tcia_index_path)[["image_id", "patient_id"]]

    # Load all histopathology tiles and map to patient IDs
    p_h_train = load_and_merge_patient_id(paths["histopathology_probabilities_train"], tcia_index)
    p_h_p003 = load_and_merge_patient_id(paths["histopathology_probabilities_p003"], tcia_index)
    p_h_p004 = load_and_merge_patient_id(paths["histopathology_probabilities_p004"], tcia_index)
    p_h_all = pd.concat([p_h_train, p_h_p003, p_h_p004], ignore_index=True)

    # Load all radiograph probabilities
    p_r_train = pd.read_csv(paths["radiograph_probabilities_train"])
    p_r_p003 = pd.read_csv(paths["radiograph_probabilities_p003"])
    p_r_p004 = pd.read_csv(paths["radiograph_probabilities_p004"])
    p_r_all = pd.concat([p_r_train, p_r_p003, p_r_p004], ignore_index=True)

    patients = sorted([p for p in p_h_all["patient_id"].unique() if p != "UNKNOWN"])
    print(f"Starting Leave-One-Patient-Out Cross-Validation on patients: {patients}")

    fold_results = {}
    for held_out_patient in patients:
        print(f"\n--- Fold: Holding out {held_out_patient} ---")

        # Split histopathology based on held-out patient
        h_train = p_h_all[p_h_all["patient_id"] != held_out_patient]
        h_test = p_h_all[p_h_all["patient_id"] == held_out_patient]

        # Prepare tensors for training
        p_h_train_tensor = torch.tensor(h_train[PROB_COLS].to_numpy(dtype="float32"))
        p_r_train_tensor = torch.tensor(p_r_all[PROB_COLS].to_numpy(dtype="float32"))
        labels_train = torch.tensor(h_train["label"].to_numpy(dtype="int64"))

        # Truncate/align training sets
        n_train = min(len(p_h_train_tensor), len(p_r_train_tensor))
        p_h_train_tensor = p_h_train_tensor[:n_train].to(device)
        p_r_train_tensor = p_r_train_tensor[:n_train].to(device)
        labels_train = labels_train[:n_train].to(device)

        # Compute optimal alpha targets for this fold's training set
        alpha_star = compute_pseudo_optimal_alpha(
            p_r_train_tensor,
            p_h_train_tensor,
            labels_train,
            default_alpha=cfg["fusion"].get("default_alpha_for_ties", 0.25),
            step=cfg["fusion"].get("alpha_grid_step", 0.05),
        )

        # Initialize gate for this fold
        gate = AdaptiveGatingNetwork(
            input_dim=cfg["adaptive_gate"].get("input_dim", 6),
            hidden_dims=cfg["adaptive_gate"].get("hidden_dims", [128, 64]),
        ).to(device)
        optimizer = torch.optim.Adam(gate.parameters(), lr=cfg["adaptive_gate"].get("learning_rate", 1e-3))
        criterion = torch.nn.MSELoss()

        dataset = TensorDataset(p_r_train_tensor, p_h_train_tensor, alpha_star)
        loader = DataLoader(dataset, batch_size=cfg["adaptive_gate"].get("batch_size", 128), shuffle=True)

        # Train gate
        gate.train()
        for epoch in range(cfg["adaptive_gate"].get("epochs", 20)):
            for batch_p_r, batch_p_h, batch_alpha in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = gate(batch_p_r, batch_p_h)
                loss = criterion(pred, batch_alpha)
                loss.backward()
                optimizer.step()

        # Evaluate on the held-out patient
        gate.eval()
        p_h_test_tensor = torch.tensor(h_test[PROB_COLS].to_numpy(dtype="float32"))
        # Align test set with a subset of the radiograph pool
        n_test = min(len(p_h_test_tensor), len(p_r_all))
        p_h_test_tensor = p_h_test_tensor[:n_test].to(device)
        p_r_test_tensor = torch.tensor(p_r_all[PROB_COLS].to_numpy(dtype="float32")[:n_test]).to(device)
        y_test = h_test["label"].to_numpy(dtype="int64")[:n_test]

        with torch.no_grad():
            alpha = gate(p_r_test_tensor, p_h_test_tensor)
            fused = adaptive_fusion(p_r_test_tensor, p_h_test_tensor, alpha).cpu().numpy()
            alpha_np = alpha.cpu().numpy().reshape(-1)

        pred = probabilities_to_predictions(fused)
        metrics = compute_classification_metrics(y_test, pred, fused)
        metrics.update({
            "n": int(n_test),
            "mean_alpha": float(np.mean(alpha_np)),
            "sd_alpha": float(np.std(alpha_np, ddof=1)) if n_test > 1 else 0.0,
        })
        fold_results[held_out_patient] = metrics
        print(f"Fold Accuracy for {held_out_patient}: {metrics['accuracy']:.4f}")

    # Compile LOPO summary
    summary = {
        "status": "completed",
        "folds": fold_results,
        "average": {
            metric: float(np.mean([fold_results[p][metric] for p in patients if fold_results[p][metric] is not None]))
            for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auroc"]
        }
    }

    out_path = ensure_dir("results/metrics") / "lopo_workflow_summary.json"
    save_json(summary, out_path)
    print(f"\nSaved LOPO summary to {out_path}")
    print("\n--- LOPO Cross-Validation Results Summary ---")
    print(f"Average Accuracy:        {summary['average']['accuracy']:.4f}")
    print(f"Average Macro F1-Score:  {summary['average']['macro_f1']:.4f}")
    print(f"Average Macro AUROC:     {summary['average']['macro_auroc']:.4f}")


if __name__ == "__main__":
    main()
