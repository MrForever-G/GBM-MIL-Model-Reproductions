📘 GBM-MIL-Model-Reproductions

Reproducible Implementations of GBM Whole Slide Image Classification Models

This repository provides clean, fully reproducible, and mutually comparable implementations of multiple WSI-based machine learning models for Glioblastoma (GBM) subtype classification.

All reproduced models follow a unified training and evaluation ecosystem, ensuring fair comparison and research-grade reproducibility.

🚀 Features
✔ Unified patient-level evaluation

Patient-level ensemble (slide-level for GBM)

Bootstrap AUC (95% CI, 1000 samples)

Multi-class ROC-AUC (macro / OVR)

Identical train/test split for all models

✔ Unified data preprocessing

DINOv2 patch embeddings (768-d)

WSI → patch extraction → feature embedding → MIL bag or graph

✔ Unified training framework

TrainProcess/YYYY-MM-DD-HH-MM-SS/ directory

TensorBoard logging

JSON experiment configurations

Identical log format for all reproduced models

✔ Unified code style

Simple, project-style comments (# ...)

Clean modular structure

Each model is an independent minimal project

No unused code or legacy leftovers

📁 Repository Structure
GBM-MIL-Model-Reproductions/
│
├── GCN-GBM-Classification/        # Graph-based WSI model (main model)
├── CLAM-GBM-Classification/        # Multi-Branch Attention MIL (reproduced)
├── TransMIL-GBM-Classification/    # coming soon
├── DSMIL-GBM-Classification/       # coming soon
├── ABMIL-GBM-Classification/       # coming soon
├── HIPT-GBM-Classification/        # coming soon
│
├── README.md                       # (this file)
└── LICENSE                         # MIT License


Each subfolder is a complete and independent model project, sharing the same:

Data format

Training loop structure

Evaluation pipeline

Logging directory layout

📚 Implemented Models
🔹 1. TwinsGCN-GBM-Classification

Graph-based WSI classifier

Patch clustering → graph construction → dual GCN tower

Ideal for spatial-aware WSI reasoning

Repository: GCN-GBM-Classification/

🔹 2. CLAM-MB (Multi-Branch MIL)

Official CLAM architecture adapted to DINOv2 patch embeddings

Multi-branch attention pooling per class

Clean reimplementation with patient-level AUC

Repository: CLAM-GBM-Classification/

⏳ Coming Next

TransMIL (Transformer MIL for whole-slide images)

DSMIL (Dual-stream MIL)

ABMIL / Ilse Attention MIL

HIPT (Hierarchical Image Pyramid Transformer)

Prov-GigaPath Feature + Graph Models

These models will be added step-by-step, each in its own folder.

📊 Unified Evaluation (very important for fair comparison)

All models output:

sampleID, y_true, y_probs


Evaluation is performed by the shared evaluation.py, including:

patient-level ensemble

bootstrap AUC (1000 resamples)

deterministic logging

compatibility with graph & MIL models

This allows the repository to produce clean tables like：

Model	AUC (95% CI)	Notes
TwinsGCN	0.xx (0.xx–0.xx)	graph-based
CLAM-MB	0.xx (0.xx–0.xx)	patch-level MIL
TransMIL	(after reproduction)	transformer MIL
📜 License

This repository is released under the MIT License (see LICENSE).

You may use, modify, distribute, or build upon this work freely, including in academic and commercial projects.

🧩 Citation

If this repository or any reproduced models are useful in your research, please cite:

@misc{gpy2025gbmmodels,
  title   = {GBM-MIL-Model-Reproductions},
  author  = {GPY (MrForever-G)},
  year    = {2025},
  note    = {Reproduced models for GBM whole-slide image classification}
}

📬 Contact

For questions, feedback, or collaboration:

GitHub: MrForever-G