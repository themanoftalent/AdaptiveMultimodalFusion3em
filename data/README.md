# Data folder

This repository intentionally does not include raw image data.

Public datasets must be downloaded from their original sources:

1. TCIA Osteosarcoma Tumor Assessment histopathology dataset.
2. Kaggle Bone Tumor Classification radiograph dataset.

The Hitit University Faculty of Medicine cohort is restricted and must not be committed to this repository.

Recommended local-only layout:

```text
data/raw/
├── tcia_osteosarcoma/
│   ├── P001/
│   ├── P002/
│   ├── P003/
│   └── P004/
├── kaggle_bone_tumor/
│   ├── Other/
│   └── Tumor/
└── restricted/
    └── hitit_authorized_only/
```

The `.gitignore` file excludes raw, private, restricted, and Hitit-related data folders.
