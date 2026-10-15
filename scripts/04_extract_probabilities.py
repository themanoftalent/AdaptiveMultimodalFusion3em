from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.efficientnet_branch import OsteoEfficientNetB0
from src.utils.datasets import ImageCSVDataset
from src.utils.io import load_yaml, ensure_dir, set_seed, resolve_device
from src.utils.transforms import build_transforms


def extract_for_csv(model, csv_path, output_csv, modality, device, input_size=224):
    transform = build_transforms(input_size, modality=modality, train=False)
    dataset = ImageCSVDataset(csv_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    rows = []
    model.eval()
    with torch.no_grad():
        for images, labels, image_ids in loader:
            images = images.to(device)
            probs = F.softmax(model(images), dim=1).cpu().numpy()
            for j, image_id in enumerate(image_ids):
                row = {"image_id": image_id, "label": int(labels[j].item())}
                if probs.shape[1] == 3:
                    row.update({"prob_non_tumor": probs[j, 0], "prob_viable_tumor": probs[j, 1], "prob_necrotic_tumor": probs[j, 2]})
                else:
                    row.update({"prob_other": probs[j, 0], "prob_tumor": probs[j, 1]})
                rows.append(row)
    ensure_dir(Path(output_csv).parent)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved probabilities to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract branch probability vectors.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--modality", choices=["histopathology", "radiograph"], required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("project", {}).get("seed", 42))
    device = resolve_device(cfg.get("project", {}).get("device", "auto"))

    num_classes = 3 if args.modality == "histopathology" else 2
    model = OsteoEfficientNetB0(num_classes=num_classes, pretrained=False).to(device)
    checkpoint = Path(cfg["paths"]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    out_dir = Path(cfg["paths"].get("probabilities_dir", "results/probabilities"))
    if args.modality == "histopathology":
        for key, name in [("train_csv", "histopathology_train_p001_p002.csv"), ("val_csv", "histopathology_p003.csv"), ("test_csv", "histopathology_p004.csv")]:
            if Path(cfg["paths"][key]).exists():
                extract_for_csv(model, cfg["paths"][key], out_dir / name, args.modality, device, cfg["preprocessing"].get("input_size", 224))
    else:
        for key, name in [("train_csv", "radiograph_train.csv"), ("val_csv", "radiograph_validation.csv"), ("test_csv", "radiograph_test.csv")]:
            if Path(cfg["paths"][key]).exists():
                extract_for_csv(model, cfg["paths"][key], out_dir / name, args.modality, device, cfg["preprocessing"].get("input_size", 224))


if __name__ == "__main__":
    main()
