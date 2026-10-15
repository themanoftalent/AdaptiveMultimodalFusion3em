from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import json
import pandas as pd

from src.utils.io import ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate result tables from JSON metric files.")
    parser.add_argument("--config", default="configs/fusion_gate.yaml")
    args = parser.parse_args()
    _ = load_yaml(args.config)
    rows = []
    for path in Path("results/metrics").glob("*.json"):
        if "lopo" in path.name:
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            for model, metrics in data.items():
                row = {"source_file": path.name, "model": model}
                row.update(metrics)
                rows.append(row)
        else:
            row = {"source_file": path.name, "model": path.stem}
            row.update(data)
            rows.append(row)
    if not rows:
        print("No metrics rows to generate.")
        return
    table = pd.DataFrame(rows)
    # Drop columns that are completely empty (all NaN)
    table = table.dropna(how='all', axis=1)
    
    out = ensure_dir("results/tables") / "metrics_summary.csv"
    table.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
