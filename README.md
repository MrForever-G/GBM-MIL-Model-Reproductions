# GBM-MIL-Model-Reproductions

A unified collection of independently implemented models for **GBM (Glioblastoma) whole-slide image classification**, including graph-based and MIL-based frameworks.  
All models follow **the same data split, evaluation pipeline, patient-level bootstrap AUC, logging structure, and coding style**, ensuring fair comparison and reproducibility.

---

## 📚 Included Models

| Model | Folder | Description |
|-------|--------|-------------|
| **TwinsGCN** | `GCN-GBM-Classification/` | Graph-based WSI classifier using cluster centers as nodes. |
| **CLAM-MB** | `CLAM-GBM-Classification/` | Multi-branch attention MIL adapted to DINOv2 patch embeddings. |
| **TransMIL** | (coming soon) | Transformer-based MIL for gigapixel pathology slides. |
| **DSMIL** | (coming soon) | Dual-stream MIL with instance selection. |
| **ABMIL** | (coming soon) | Attention-based MIL (Ilse et al., 2018). |
| **HIPT / ViT-Large** | (coming soon) | Hierarchical image pyramid transformer. |

---

## 🧬 Unified Evaluation Pipeline

All models share the following components:

- **train/test split**: identical across all methods  
- **DINOv2 patch embeddings** (768-d)  
- **patient-level aggregation** (slide-level for GBM)  
- **bootstrap AUC (95% CI)**  
- **TrainProcess/ logging structure**  
- **TensorBoard visualization**  
- **consistent annotation style and clean project layout**

---

## 📁 Repository Structure
GBM-MIL-Model-Reproductions/
│
├── GCN-GBM-Classification/
├── CLAM-GBM-Classification/
├── TransMIL-GBM-Classification/ (future)
├── DSMIL-GBM-Classification/ (future)
├── ABMIL-GBM-Classification/ (future)
├── HIPT-GBM-Classification/ (future)
│
├── README.md
└── LICENSE


---

## 📝 Citation

If you use this repository or any reproduced models, please cite:



@misc{gpy2025gbmmodels,
title = {GBM-MIL-Model-Reproductions},
author = {GPY (MrForever-G)},
year = {2025},
note = {Reproducible implementations of GBM classification models}
}


---

## 📜 License

This project is licensed under the **MIT License** (see `LICENSE`).

---

## 📧 Contact

For questions, discussions or collaboration:  
**MrForever-G on GitHub**
