from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL_MAP = {
    "non tumor": 0,
    "non-tumor": 0,
    "nontumor": 0,
    "viable tumor": 1,
    "viable": 1,
    "necrotic tumor": 2,
    "necrotic": 2,
}

LABEL_NAME = {0: "Non tumor", 1: "Viable tumor", 2: "Necrotic tumor"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def infer_label(path: Path) -> int | None:
    text = " ".join(part.lower().replace("_", "-") for part in path.parts)
    for key, value in LABEL_MAP.items():
        if key in text.replace("-", " "):
            return value
    return None


def infer_patient(path: Path) -> str:
    for part in path.parts:
        upper = part.upper()
        if upper in {"P001", "P002", "P003", "P004"}:
            return upper
    return "UNKNOWN"


def build_tcia_index(root_dir: str | Path, output_csv: str | Path) -> pd.DataFrame:
    root_dir = Path(root_dir)
    rows = []
    for image_path in sorted(root_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = infer_label(image_path)
        if label is None:
            continue
        patient_id = infer_patient(image_path)
        split = "train" if patient_id in {"P001", "P002"} else "validation" if patient_id == "P003" else "test" if patient_id == "P004" else "unknown"
        rows.append({
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "label": label,
            "label_name": LABEL_NAME[label],
            "patient_id": patient_id,
            "split": split,
            "source": "TCIA",
        })
    frame = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def write_patient_split_files(index_csv: str | Path, output_dir: str | Path) -> None:
    frame = pd.read_csv(index_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame[frame["patient_id"].isin(["P001", "P002"])].to_csv(output_dir / "tcia_train_p001_p002.csv", index=False)
    frame[frame["patient_id"] == "P003"].to_csv(output_dir / "tcia_p003_validation.csv", index=False)
    frame[frame["patient_id"] == "P004"].to_csv(output_dir / "tcia_p004_test.csv", index=False)
