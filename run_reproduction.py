from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public-data reproduction workflow.")
    parser.add_argument("--paths-config", default="configs/paths_template.yaml")
    parser.add_argument("--skip-training", action="store_true", help="Use existing probability files and skip branch training.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    python = sys.executable

    run([python, "scripts/01_prepare_data.py", "--config", args.paths_config])

    if not args.skip_training:
        run([python, "scripts/02_train_histopathology.py", "--config", "configs/histopathology_efficientnet_b0.yaml"])
        run([python, "scripts/03_train_radiograph.py", "--config", "configs/radiograph_efficientnet_b0.yaml"])
        run([python, "scripts/04_extract_probabilities.py", "--config", "configs/histopathology_efficientnet_b0.yaml", "--modality", "histopathology"])
        run([python, "scripts/04_extract_probabilities.py", "--config", "configs/radiograph_efficientnet_b0.yaml", "--modality", "radiograph"])

    for script, cfg in [
        ("scripts/05_remap_radiograph_probabilities.py", "configs/fusion_gate.yaml"),
        ("scripts/06_train_gate.py", "configs/fusion_gate.yaml"),
        ("scripts/07_run_baselines.py", "configs/baselines.yaml"),
        ("scripts/08_evaluate_p004.py", "configs/fusion_gate.yaml"),
        ("scripts/09_evaluate_lopo.py", "configs/fusion_gate.yaml"),
        ("scripts/11_generate_tables.py", "configs/fusion_gate.yaml"),
        ("scripts/12_generate_figures.py", "configs/fusion_gate.yaml"),
    ]:
        run([python, script, "--config", cfg])


if __name__ == "__main__":
    main()
