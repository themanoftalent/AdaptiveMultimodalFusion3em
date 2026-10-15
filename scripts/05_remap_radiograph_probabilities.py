from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.fusion.remap_radiograph_probabilities import remap_probability_csv
from src.utils.io import load_yaml, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap binary radiograph probabilities to three-class prior.")
    parser.add_argument("--config", default="configs/fusion_gate.yaml")
    args = parser.parse_args()
    _ = load_yaml(args.config)
    ensure_dir("results/probabilities")
    mapping = {
        "results/probabilities/radiograph_train.csv": "results/probabilities/radiograph_train_remapped.csv",
        "results/probabilities/radiograph_validation.csv": "results/probabilities/radiograph_p003_remapped.csv",
        "results/probabilities/radiograph_test.csv": "results/probabilities/radiograph_p004_remapped.csv",
    }
    for src, dst in mapping.items():
        if Path(src).exists():
            remap_probability_csv(src, dst)
            print(f"Saved {dst}")
        else:
            print(f"Skipping missing file: {src}")


if __name__ == "__main__":
    main()
