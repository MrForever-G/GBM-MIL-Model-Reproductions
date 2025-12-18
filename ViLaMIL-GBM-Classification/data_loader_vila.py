# data_loader_vila.py
# Dataset for ViLa-MIL using slide-level .pt patch features + cluster features

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class ViLaMILDataset(Dataset):
    # CSV columns: sampleID, GeneExp_Subtype
    def __init__(self, csv_path, pt_root, cluster_root):
        super().__init__()
        self.pt_root = pt_root
        self.cluster_root = cluster_root

        df = pd.read_csv(csv_path)

        # Original 4-class mapping from the main GCN project
        label_type = {
            "Classical": 0,
            "Mesenchymal": 1,
            "Neural": 2,
            "Proneural": 3,
        }

        sample_ids = []
        labels = []

        # Filter to Mesenchymal and Proneural, then map to binary labels
        for _, row in df.iterrows():
            sid = str(row["sampleID"])
            subtype = row["GeneExp_Subtype"]

            raw_label = label_type[subtype]

            # Keep only Mesenchymal (1) and Proneural (3)
            if raw_label not in [1, 3]:
                continue

            # Binary mapping: Mesenchymal -> 0, Proneural -> 1
            if raw_label == 1:
                bin_label = 0
            else:
                bin_label = 1

            sample_ids.append(sid)
            labels.append(bin_label)

        self.sample_ids = sample_ids
        self.labels = labels

    def __len__(self):
        return len(self.sample_ids)

    # Load slide-level patch features (.pt, from patch aggregation)
    def _load_patch_features(self, slide_id):
        pt_path = os.path.join(self.pt_root, f"{slide_id}.pt")

        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Patch .pt file missing: {pt_path}")

        data = torch.load(pt_path)

        feat = data["features"].numpy().astype(np.float32)   # [N, 768]
        coords = data["coords"].numpy().astype(np.float32)   # [N, 2]
        return feat, coords

    # Load cluster-level GCN features
    def _load_cluster_features(self, slide_id):
        rpt0 = os.path.join(self.cluster_root, f"{slide_id}_rpt_0.pt")
        rpt_none = os.path.join(self.cluster_root, f"{slide_id}_rpt_none.pt")

        if os.path.exists(rpt0):
            data = torch.load(rpt0)
        elif os.path.exists(rpt_none):
            data = torch.load(rpt_none)
        else:
            raise FileNotFoundError(
                f"Cluster file missing: {slide_id}_rpt_0.pt / rpt_none.pt"
            )

        center_feat = data["center_feature"].numpy().astype(np.float32)
        center_coord = data["center_corrd"].numpy().astype(np.float32)
        return center_feat, center_coord

    def __getitem__(self, idx):
        slide_id = self.sample_ids[idx]
        label = self.labels[idx]

        patch_feat, patch_coord = self._load_patch_features(slide_id)
        cluster_feat, cluster_coord = self._load_cluster_features(slide_id)

        return {
            "patch_feat": torch.from_numpy(patch_feat),
            "patch_coord": torch.from_numpy(patch_coord),
            "cluster_feat": torch.from_numpy(cluster_feat),
            "cluster_coord": torch.from_numpy(cluster_coord),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": slide_id,
        }
