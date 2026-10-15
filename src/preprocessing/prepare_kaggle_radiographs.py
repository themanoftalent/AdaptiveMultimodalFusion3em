from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def infer_radiograph_label(path: Path) -> tuple[int, str]:
    text = " ".join(part.lower() for part in path.parts)
    if "osteosarcoma" in text or "tumor" in text:
        return 1, "Tumor"
    return 0, "Other"


def build_kaggle_index(root_dir: str | Path, output_csv: str | Path, seed: int = 42) -> pd.DataFrame:
    root_dir = Path(root_dir)
    rows = []
    for image_path in sorted(root_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label, label_name = infer_radiograph_label(image_path)
        rows.append({
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "label": label,
            "label_name": label_name,
            "source": "Kaggle",
        })

    frame = pd.DataFrame(rows)
    if len(frame) > 0 and frame["label"].nunique() > 1:
        train_val, test = train_test_split(frame, test_size=0.2, random_state=seed, stratify=frame["label"])
        train, val = train_test_split(train_val, test_size=0.2, random_state=seed, stratify=train_val["label"])
        frame.loc[train.index, "split"] = "train"
        frame.loc[val.index, "split"] = "validation"
        frame.loc[test.index, "split"] = "test"
    else:
        frame["split"] = "unknown"

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def write_radiograph_split_files(index_csv: str | Path, output_dir: str | Path) -> None:
    frame = pd.read_csv(index_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, file_name in [("train", "kaggle_train.csv"), ("validation", "kaggle_validation.csv"), ("test", "kaggle_test.csv")]:
        frame[frame["split"] == split_name].to_csv(output_dir / file_name, index=False)
