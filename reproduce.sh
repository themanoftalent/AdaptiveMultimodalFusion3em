#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/paths_template.yaml}

echo "Running public-data reproducibility workflow."
echo "Config: ${CONFIG_PATH}"
echo "Restricted Hitit validation is not run by this script."

python scripts/01_prepare_data.py --config "${CONFIG_PATH}"
python scripts/02_train_histopathology.py --config configs/histopathology_efficientnet_b0.yaml
python scripts/03_train_radiograph.py --config configs/radiograph_efficientnet_b0.yaml
python scripts/04_extract_probabilities.py --config configs/histopathology_efficientnet_b0.yaml --modality histopathology
python scripts/04_extract_probabilities.py --config configs/radiograph_efficientnet_b0.yaml --modality radiograph
python scripts/05_remap_radiograph_probabilities.py --config configs/fusion_gate.yaml
python scripts/06_train_gate.py --config configs/fusion_gate.yaml
python scripts/07_run_baselines.py --config configs/baselines.yaml
python scripts/08_evaluate_p004.py --config configs/fusion_gate.yaml
python scripts/09_evaluate_lopo.py --config configs/fusion_gate.yaml
python scripts/11_generate_tables.py --config configs/fusion_gate.yaml
python scripts/12_generate_figures.py --config configs/fusion_gate.yaml

echo "Public reproducibility workflow completed."
