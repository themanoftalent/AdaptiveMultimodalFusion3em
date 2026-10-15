from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.efficientnet_branch import OsteoEfficientNetB0
from src.utils.datasets import ImageCSVDataset
from src.utils.io import load_yaml, ensure_dir, set_seed, resolve_device
from src.utils.transforms import build_transforms


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
    return total_loss / max(len(loader.dataset), 1)


def validate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * images.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total_samples += images.size(0)
    return total_loss / max(total_samples, 1), correct / max(total_samples, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train histopathology EfficientNet-B0 branch.")
    parser.add_argument("--config", default="configs/histopathology_efficientnet_b0.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("project", {}).get("seed", 42))
    device = resolve_device(cfg.get("project", {}).get("device", "auto"))

    train_csv = Path(cfg["paths"]["train_csv"])
    val_csv = Path(cfg["paths"]["val_csv"])
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}. Run scripts/01_prepare_data.py first.")

    # Training Loader
    train_transform = build_transforms(cfg["preprocessing"].get("input_size", 224), modality="histopathology", train=True)
    train_dataset = ImageCSVDataset(train_csv, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg["training"].get("batch_size", 32), shuffle=True, num_workers=2)

    # Validation Loader
    val_loader = None
    if val_csv.exists():
        val_transform = build_transforms(cfg["preprocessing"].get("input_size", 224), modality="histopathology", train=False)
        val_dataset = ImageCSVDataset(val_csv, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=cfg["training"].get("batch_size", 32), shuffle=False, num_workers=2)
    else:
        print(f"Validation CSV not found at {val_csv}. Training without validation validation steps.")

    model = OsteoEfficientNetB0(num_classes=3, pretrained=cfg["model"].get("pretrained", True)).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"].get("learning_rate", 1e-4),
        betas=(cfg["training"].get("beta1", 0.9), cfg["training"].get("beta2", 0.999)),
        weight_decay=cfg["training"].get("weight_decay", 1e-5),
    )
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["training"].get("scheduler_factor", 0.5),
        patience=cfg["training"].get("scheduler_patience", 3)
    )

    early_stopping_patience = cfg["training"].get("early_stopping_patience", 10)
    epochs_no_improve = 0
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None

    print(f"Training OsteoEfficientNetB0 (histopathology) on {device}...")
    for epoch in range(cfg["training"].get("max_epochs", 50)):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch + 1:02d}/{cfg['training'].get('max_epochs', 50)}: "
                  f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_acc={val_acc:.4f}")

            # Step the scheduler
            scheduler.step(val_loss)

            # Keep track of the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement on validation loss.")
                    break
        else:
            print(f"Epoch {epoch + 1:02d}/{cfg['training'].get('max_epochs', 50)}: train_loss={train_loss:.6f}")

    if best_model_state is not None:
        print(f"Restoring best model from Epoch {best_epoch} (val_loss={best_val_loss:.6f})")
        model.load_state_dict(best_model_state)
    else:
        best_model_state = model.state_dict()

    checkpoint_path = Path(cfg["paths"]["checkpoint"])
    ensure_dir(checkpoint_path.parent)
    torch.save({"model_state_dict": best_model_state, "config": cfg}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
