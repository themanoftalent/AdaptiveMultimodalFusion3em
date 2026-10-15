from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd

from src.preprocessing.prepare_tcia_tiles import build_tcia_index, write_patient_split_files
from src.preprocessing.prepare_kaggle_radiographs import build_kaggle_index, write_radiograph_split_files
from src.utils.io import load_yaml, ensure_dir, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public dataset metadata and split files.")
    parser.add_argument("--config", default="configs/paths_template.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("project", {}).get("seed", 42))
    paths = cfg.get("paths", {})
    metadata = cfg.get("metadata", {})
    ensure_dir("data/processed")

    tcia_root = Path(paths.get("tcia_root", ""))
    kaggle_root = Path(paths.get("kaggle_root", ""))

    if tcia_root.exists():
        tcia_index = build_tcia_index(tcia_root, metadata.get("tcia_index_csv", "data/processed/tcia_index.csv"))
        write_patient_split_files(metadata.get("tcia_index_csv", "data/processed/tcia_index.csv"), "data/processed")
        print(f"Prepared TCIA index with {len(tcia_index)} images.")
    else:
        print(f"TCIA root not found: {tcia_root}. Edit configs/paths_template.yaml before running full reproduction.")

    if kaggle_root.exists():
        kaggle_index = build_kaggle_index(kaggle_root, metadata.get("kaggle_index_csv", "data/processed/kaggle_index.csv"), seed=cfg.get("project", {}).get("seed", 42))
        write_radiograph_split_files(metadata.get("kaggle_index_csv", "data/processed/kaggle_index.csv"), "data/processed")
        print(f"Prepared Kaggle index with {len(kaggle_index)} images.")
    else:
        print(f"Kaggle root not found: {kaggle_root}. Edit configs/paths_template.yaml before running full reproduction.")


if __name__ == "__main__":
    main()
