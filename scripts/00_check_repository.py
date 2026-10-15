from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from pathlib import Path

REQUIRED = [
    "README.md",
    "requirements.txt",
    "environment.yml",
    "configs/histopathology_efficientnet_b0.yaml",
    "configs/radiograph_efficientnet_b0.yaml",
    "configs/fusion_gate.yaml",
    "data/manifest_template.csv",
    "src/models/efficientnet_branch.py",
    "src/models/gating_network.py",
    "src/fusion/pseudo_optimal_alpha.py",
]


def main() -> None:
    missing = [path for path in REQUIRED if not Path(path).exists()]
    if missing:
        raise SystemExit("Missing required files: " + ", ".join(missing))
    print("Repository structure check passed.")


if __name__ == "__main__":
    main()
