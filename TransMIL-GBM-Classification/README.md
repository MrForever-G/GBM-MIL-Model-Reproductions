TransMIL-GBM-Classification
Transformer-based Multiple Instance Learning reproduction for GBM whole-slide images
Reproducible • Minimal • Research-grade

Introduction
This project provides a clean and fully reproducible implementation of TransMIL (CVPR 2021) adapted for GBM four-class classification.
The implementation follows the unified framework defined in GBM-MIL-Model-Reproductions, ensuring consistent data format, training flow, logging structure, and patient-level evaluation across all models (TwinsGCN, CLAM, TransMIL).

Project Structure

models_transmil.py TransMIL model (official architecture)
data_loader_transmil.py Loader for WSI-level MIL bags (.pt features)
train_transmil.py Training pipeline (pure PyTorch)
run_transmil.py Entry script with timestamped log directory

evaluation.py Shared patient-level evaluation module
utils.py Shared utilities (logging, etc.)

pt_files/ WSI patch embedding files
train_sample_id.csv Training split
test_sample_id.csv Testing split

TrainProcess/ Auto-created experiment folders
└── 2025-xx-xx-xx/
log.txt
best.pth
test_slide_preds.csv

Data Format

Each WSI is represented as a .pt file containing:
center_feature : Tensor [N, 768]
center_corrd : Tensor [N, 2]

CSV split format:
slide_id,label
GBM001,0
GBM002,3

slide_id must match the filenames inside pt_files.

Training

TransMIL can be launched directly with:

python run_transmil.py
--train_csv train_sample_id.csv
--test_csv test_sample_id.csv
--pt_dir pt_files
--num_classes 4
--feat_dim 768
--epochs 20
--lr 1e-4
--log_root TrainProcess

A new timestamped directory will be created under TrainProcess/, containing log.txt, best.pth and test predictions.

Testing

Testing is automatically executed at the end of training.
Results are saved to:
test_slide_preds.csv

The file contains:
slide_id
label
prob_0
prob_1
prob_2
prob_3

Evaluation

Evaluation follows the unified patient-level pipeline:

patient-level fusion
bootstrap AUC (1000×)
macro / OVR ROC-AUC
deterministic logging format

Consistent with CLAM and TwinsGCN for fair comparison.

Explainability

The model outputs:
logits
attention maps (from all transformer layers)

These can be used for:
Attention rollout
MIL attention visualization
Graph-CAM integration with TwinsGCN

Acknowledgements

TransMIL paper:
Shao et al., “TransMIL: Transformer-based MIL for Whole Slide Image Classification”, CVPR 2021.

Official implementation:
https://github.com/szc19990412/TransMIL

License

This project is released under the MIT License, consistent with GBM-MIL-Model-Reproductions.