# Reproducibility workflow

This document describes the computational path implemented in the repository.

## 1. Dataset acquisition

Download public datasets from the original sources. Do not redistribute raw images in this repository.

## 2. Dataset indexing

Run `scripts/01_prepare_data.py` to generate CSV metadata files. The TCIA CSV contains `image_path`, `label`, `label_name`, `patient_id`, and `split`. The Kaggle CSV contains `image_path`, `label`, `label_name`, and `split`.

## 3. Branch training

Train the histopathology and radiograph branches independently. This preserves the unpaired design. The radiograph branch is binary before remapping. The histopathology branch is three-class.

## 4. Probability extraction

After branch training, save frozen branch outputs as probability CSV files. Fusion scripts operate on these CSV files, not on paired images.

## 5. Radiograph remapping

Binary radiograph probabilities are remapped into the three-class histopathology space:

```text
P_non_tumor = P_other
P_viable = P_tumor / 2
P_necrotic = P_tumor / 2
```

## 6. Fusion

The repository supports fixed-alpha fusion, max-confidence selection, attention-style confidence fusion, classical baselines on concatenated probability vectors, and adaptive gated fusion.

## 7. Evaluation

P004 is treated as the primary held-out TCIA test patient. P003 is the validation/calibration patient. Leave-one-patient-out evaluation requires repeating the workflow with each patient held out.

## 8. Restricted Hitit validation

The Hitit cohort is not public. The repository includes only `scripts/10_external_hitit_interface.py` and a metadata template. Authorized users may run the interface locally, but raw images and private metadata must not be uploaded.
