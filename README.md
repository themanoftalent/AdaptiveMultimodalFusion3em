# Osteosarcoma Adaptive Fusion — Reproducibility Package

**Manuscript:** Adaptive Multimodal Fusion of Histopathology and Radiographs for Osteosarcoma Classification Using EfficientNet and Gated Decision Integration

---

> **Repository notice:** This repository is provided for peer-review and reproducibility inspection. It does not grant permission for reuse, redistribution, modification, or commercial use of the code without written permission from the authors. Raw image data and restricted institutional data are not included. See LICENSE for full terms.

---

## What this repository provides

The repository provides the implementation and documentation required to reproduce the public-data computational workflow after users obtain the TCIA and Kaggle datasets from their original source platforms. Raw image data and restricted institutional data are not included. The repository documents the preprocessing, model configuration, probability extraction, radiograph probability remapping, decision-level fusion, baseline evaluation, and figure/table generation steps used in the revised paper.

Independent researchers can inspect and reproduce the public-data workflow after downloading the public datasets.

---

## Important data and scope statements

- **No raw image data are included.** Raw TCIA histopathology tiles and Kaggle radiographs must be downloaded by the user from their original platforms.
- **The TCIA Osteosarcoma Tumor Assessment dataset and the Kaggle Bone Tumor Classification dataset must be obtained independently** from The Cancer Imaging Archive and Kaggle respectively.
- **Hitit University data are restricted and are not redistributed.** The Hitit cohort is subject to institutional ethics approval and data-governance conditions. The repository provides only an evaluation script interface and a metadata template for use by authorized institutions.
- **The implementation does not assume paired images across modalities.** The histopathology and radiograph branches are trained independently on separate datasets. No image-level pairing between TCIA tiles and Kaggle radiographs is assumed or required.
- **Fusion operates on saved probability vectors, not on raw images.** After each branch produces frozen softmax probability vectors, all fusion, baseline, and evaluation steps use only those saved CSV files.

---

## Workflow summary

```
TCIA histopathology tiles  →  EfficientNet-B0 (3-class)  →  probability vectors
                                                                      │
Kaggle radiographs         →  EfficientNet-B0 (2-class)  →  probability vectors
                                                               ↓
                                              binary→three-class remapping
                                                               ↓
                                         decision-level fusion on probability vectors
                                    ┌──────────────────┬──────────────────┐
                               fixed-alpha         max-confidence    adaptive gating MLP
                                    └──────────────────┴──────────────────┘
                                                               ↓
                                         baselines: XGBoost, LogReg, SVM, MLP, Attention
                                                               ↓
                                         P004 held-out test + LOPO cross-validation
                                                               ↓
                                         Hitit external validation (authorized use only)
```

---

## Repository structure

```text
osteosarcoma-adaptive-fusion-reproducibility/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── reproduce.sh                          ← one-command public workflow
├── configs/
│   ├── paths_template.yaml               ← edit local dataset paths here
│   ├── histopathology_efficientnet_b0.yaml
│   ├── radiograph_efficientnet_b0.yaml
│   ├── fusion_gate.yaml
│   └── baselines.yaml
├── data/
│   ├── README.md
│   ├── manifest_template.csv
│   ├── hitit_metadata.csv                ← governance notice only, no patient data
│   └── splits/
│       └── tcia_patient_splits.csv       ← patient-level split assignment (P001–P004)
├── docs/
│   ├── reproducibility_workflow.md
│   ├── data_manifest.md
│   └── known_limitations.md
├── figures/                              ← populated after running scripts/12
├── results/                              ← intentionally empty; populated by workflow
│   ├── probabilities/
│   ├── metrics/
│   └── tables/
├── scripts/
│   ├── 00_check_repository.py
│   ├── 01_prepare_data.py
│   ├── 02_train_histopathology.py
│   ├── 03_train_radiograph.py
│   ├── 04_extract_probabilities.py
│   ├── 05_remap_radiograph_probabilities.py
│   ├── 06_train_gate.py
│   ├── 07_run_baselines.py
│   ├── 08_evaluate_p004.py
│   ├── 09_evaluate_lopo.py
│   ├── 10_external_hitit_interface.py
│   ├── 11_generate_tables.py
│   └── 12_generate_figures.py
├── src/
│   ├── models/
│   │   ├── efficientnet_branch.py        ← EfficientNet-B0 for both modalities
│   │   └── gating_network.py             ← adaptive gating MLP (6-128-64-1)
│   ├── fusion/
│   │   ├── adaptive_gate.py
│   │   ├── fixed_alpha.py
│   │   ├── max_confidence.py
│   │   ├── pseudo_optimal_alpha.py
│   │   └── remap_radiograph_probabilities.py
│   ├── baselines/
│   │   ├── xgboost_probability_baseline.py
│   │   ├── logistic_regression.py
│   │   ├── linear_svm.py
│   │   ├── shallow_mlp.py
│   │   └── attention_late_fusion.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── bootstrap_ci.py
│   │   ├── calibration.py
│   │   └── mcnemar.py
│   ├── preprocessing/
│   │   ├── macenko_normalization.py
│   │   ├── prepare_tcia_tiles.py
│   │   └── prepare_kaggle_radiographs.py
│   └── utils/
│       ├── datasets.py
│       ├── transforms.py
│       └── io.py
└── supplementary/
    └── reproducibility_workflow_and_experimental_configuration.md
```

---

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate osteo-fusion-repro
```

Or use pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step 1 — Obtain the public datasets

**TCIA Osteosarcoma Tumor Assessment**
Download from The Cancer Imaging Archive. The dataset contains whole-slide image tiles for four patients (P001–P004) with three histopathology classes: Non tumor, Viable tumor, Necrotic tumor.

**Kaggle Bone Tumor Classification**
Download from the Kaggle dataset page. The dataset contains radiograph images with two classes: Other and Tumor.

Place downloaded data under local paths that are not committed to GitHub. A recommended layout:

```text
/data/osteosarcoma/
├── tcia_osteosarcoma/
│   ├── P001/
│   ├── P002/
│   ├── P003/
│   └── P004/
└── kaggle_bone_tumor/
    ├── Other/
    └── Tumor/
```

Edit the local paths in `configs/paths_template.yaml` before running any script.

---

## Step 2 — Prepare data indices and split files

```bash
python scripts/01_prepare_data.py --config configs/paths_template.yaml
```

This script indexes image paths under the configured directories and writes split CSV files to `data/processed/` and `data/splits/`. The patient-level split assignment (`data/splits/tcia_patient_splits.csv`) is already committed to the repository: P001 and P002 are training, P003 is validation and calibration, P004 is the primary held-out test set.

---

## Step 3 — Train the histopathology branch

```bash
python scripts/02_train_histopathology.py --config configs/histopathology_efficientnet_b0.yaml
```

EfficientNet-B0 pretrained on ImageNet, last two feature blocks unfrozen, three-class output head. Input size 224 × 224. Macenko stain normalisation. Adam optimizer, learning rate 0.0001, weight decay 1e-5, batch size 32, early stopping patience 10, ReduceLROnPlateau factor 0.5 patience 3. Random seed 42.

---

## Step 4 — Train the radiograph branch

```bash
python scripts/03_train_radiograph.py --config configs/radiograph_efficientnet_b0.yaml
```

Same EfficientNet-B0 architecture, two-class output head (Other / Tumor). Min-max normalisation. Same optimizer and scheduler settings as the histopathology branch.

---

## Step 5 — Extract frozen probability vectors

```bash
python scripts/04_extract_probabilities.py --config configs/histopathology_efficientnet_b0.yaml --modality histopathology
python scripts/04_extract_probabilities.py --config configs/radiograph_efficientnet_b0.yaml --modality radiograph
```

Both branches are frozen after training. Softmax probabilities for every image in every split are written to `results/probabilities/`. All subsequent steps use only these CSV files. No raw images are read after this point.

---

## Step 6 — Remap radiograph probabilities

```bash
python scripts/05_remap_radiograph_probabilities.py --config configs/fusion_gate.yaml
```

The radiograph branch produces two-class probabilities (P_other, P_tumor). These are remapped into the three-class histopathology space before fusion:

```
P_non_tumor  = P_other
P_viable     = P_tumor / 2
P_necrotic   = P_tumor / 2
```

This remapping is intentionally simple and transparent. It allows the radiograph branch to contribute to the shared three-class decision space without implying that the radiograph labels are equivalent to the histopathology labels.

---

## Step 7 — Train the adaptive gating network

```bash
python scripts/06_train_gate.py --config configs/fusion_gate.yaml
```

The gating MLP takes the concatenated six-dimensional probability vector `[P_radiograph ; P_histopathology]` as input and outputs a sample-specific fusion weight alpha. Architecture: input 6 → hidden 128 → hidden 64 → sigmoid output. Trained with MSE loss against pseudo-optimal alpha targets derived from a grid search over alpha in [0, 1] with step 0.05. Fixed-alpha default is 0.25.

---

## Step 8 — Run baseline comparisons

```bash
python scripts/07_run_baselines.py --config configs/baselines.yaml
```

Baselines evaluated on the same frozen probability vectors:

- Fixed-alpha late fusion (alpha = 0.25)
- Max-confidence decision selection
- XGBoost
- Logistic regression
- Linear SVM
- Shallow MLP
- Attention-style late fusion

---

## Step 9 — Evaluate on the held-out test set and LOPO folds

```bash
python scripts/08_evaluate_p004.py --config configs/fusion_gate.yaml
python scripts/09_evaluate_lopo.py --config configs/fusion_gate.yaml
```

P004 is the primary held-out test set. Leave-one-patient-out cross-validation uses all four TCIA patients. Metrics include accuracy, macro F1, macro AUROC, calibration (ECE, Brier score), and McNemar test against the strongest baseline.

---

## Step 10 — External validation (authorized use only)

```bash
python scripts/10_external_hitit_interface.py \
  --metadata data/hitit_metadata.csv \
  --predictions results/probabilities/hitit_predictions_authorized.csv
```

This script accepts externally supplied prediction files for the Hitit cohort. Hitit data are not included in this repository and cannot be redistributed. The script is provided so that authorized institutions can run the same evaluation protocol against their local copy of the data.

---

## Step 11 — Generate tables and figures

```bash
python scripts/11_generate_tables.py --config configs/fusion_gate.yaml
python scripts/12_generate_figures.py --config configs/fusion_gate.yaml
```

---

## One-command public workflow

After completing Steps 1 and 2 (dataset download and path configuration):

```bash
bash reproduce.sh
```

This runs Steps 2 through 11 in order. Hitit external validation is not included.

---

## Results directory

The `results/` directory is intentionally empty in the public repository. It is populated only after the user downloads the required public datasets and runs the reproduction workflow. No mock, synthetic, or placeholder results are included.

---

## Class labels

| Branch | Class index | Class name |
|---|---|---|
| Histopathology | 0 | Non tumor |
| Histopathology | 1 | Viable tumor |
| Histopathology | 2 | Necrotic tumor |
| Radiograph (pre-remap) | 0 | Other |
| Radiograph (pre-remap) | 1 | Tumor |

---

## Key hyperparameters (both branches)

| Parameter | Value |
|---|---|
| Architecture | EfficientNet-B0 |
| Pretrained weights | ImageNet1K |
| Input size | 224 × 224 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Beta1 / Beta2 | 0.9 / 0.999 |
| Weight decay | 1e-5 |
| Batch size | 32 |
| Early stopping patience | 10 epochs |
| LR scheduler | ReduceLROnPlateau, factor 0.5, patience 3 |
| Random seed | 42 |

| Gating MLP parameter | Value |
|---|---|
| Input dimension | 6 |
| Hidden layers | 128, 64 |
| Output activation | Sigmoid |
| Loss function | MSE |
| Alpha grid step | 0.05 |
| Fixed-alpha baseline | 0.25 |

---

## Known limitations

1. The public TCIA histopathology cohort contains only four patients. Statistical conclusions from this cohort alone are limited by patient-level diversity.
2. The public radiographs and histopathology tiles are unpaired. Fusion is performed on probability vectors to preserve this condition.
3. The radiograph probability remapping is heuristic. The two-class radiograph labels are not equivalent to the three histopathology classes.
4. The Hitit cohort cannot be redistributed. Only the evaluation interface and metadata template are included for authorized use.
5. Exact numerical reproduction requires the same public dataset versions, image preprocessing settings, library versions, and random seed. Minor floating-point differences between hardware and library versions are expected.

---

## What is not included

- Raw TCIA histopathology tile images
- Raw Kaggle radiograph images
- Hitit University images or patient records
- Pre-computed probability vectors or metrics

---

## ☕ Sponsor or Buy Me a Coffee

If you find this project helpful, consider supporting it or reaching out for collaboration:

[![Buy Me A Coffee](https://img.shields.io/badge/BUY%20ME%20A%20COFFEE-SUPPORT%20PROJECT-green?style=for-the-badge&logo=buymeacoffee)](https://buymeacoffee.com/themanoftalent) [![GitHub Sponsors](https://img.shields.io/badge/SPONSOR-GITHUB%20SPONSORS-ea4aaa?style=for-the-badge&logo=githubsponsors)](https://github.com/sponsors/themanoftalent) [![Let's Connect](https://img.shields.io/badge/LET%27S%20CONNECT-EMAIL-blue?style=for-the-badge&logo=gmail)](mailto:cs.akifcifci@gmail.com)
