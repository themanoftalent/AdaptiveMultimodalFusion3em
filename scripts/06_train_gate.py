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

from src.fusion.pseudo_optimal_alpha import compute_pseudo_optimal_alpha
from src.models.gating_network import AdaptiveGatingNetwork
from src.utils.io import load_yaml, ensure_dir, set_seed, resolve_device

PROB_COLS = ["prob_non_tumor", "prob_viable_tumor", "prob_necrotic_tumor"]


def load_probabilities(path):
    frame = pd.read_csv(path)
    return torch.tensor(frame[PROB_COLS].to_numpy(dtype="float32")), frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Train adaptive gate on frozen probability vectors.")
    parser.add_argument("--config", default="configs/fusion_gate.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("project", {}).get("seed", 42))
    device = resolve_device(cfg.get("project", {}).get("device", "auto"))
    paths = cfg["paths"]

    p_h, frame_h = load_probabilities(paths["histopathology_probabilities_train"])
    p_r, frame_r = load_probabilities(paths["radiograph_probabilities_train"])
    labels = torch.tensor(frame_h["label"].to_numpy(dtype="int64"))

    n = min(len(p_h), len(p_r), len(labels))
    p_h, p_r, labels = p_h[:n].to(device), p_r[:n].to(device), labels[:n].to(device)
    alpha_star = compute_pseudo_optimal_alpha(
        p_r,
        p_h,
        labels,
        default_alpha=cfg["fusion"].get("default_alpha_for_ties", 0.25),
        step=cfg["fusion"].get("alpha_grid_step", 0.05),
    )

    gate = AdaptiveGatingNetwork(
        input_dim=cfg["adaptive_gate"].get("input_dim", 6),
        hidden_dims=cfg["adaptive_gate"].get("hidden_dims", [128, 64]),
    ).to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=cfg["adaptive_gate"].get("learning_rate", 1e-3))
    criterion = torch.nn.MSELoss()
    dataset = TensorDataset(p_r, p_h, alpha_star)
    loader = DataLoader(dataset, batch_size=cfg["adaptive_gate"].get("batch_size", 128), shuffle=True)

    for epoch in range(cfg["adaptive_gate"].get("epochs", 20)):
        total = 0.0
        for batch_p_r, batch_p_h, batch_alpha in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = gate(batch_p_r, batch_p_h)
            loss = criterion(pred, batch_alpha)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_p_r)
        print(f"Epoch {epoch + 1}: gate_mse={total / max(len(dataset), 1):.6f}")

    checkpoint_path = Path(paths["gate_checkpoint"])
    ensure_dir(checkpoint_path.parent)
    torch.save({"model_state_dict": gate.state_dict(), "config": cfg, "parameter_count": gate.count_parameters()}, checkpoint_path)
    print(f"Saved adaptive gate to {checkpoint_path}; trainable parameters = {gate.count_parameters()}")


if __name__ == "__main__":
    main()
