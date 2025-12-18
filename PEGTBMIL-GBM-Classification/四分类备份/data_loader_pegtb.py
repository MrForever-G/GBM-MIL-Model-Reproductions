# data_loader_pegtb.py
# Dataset for PEGTB-MIL with coordinate-guided attention (binary setting)
# Input:  slide-level {features, coords} with shape [1, N, ...]
# Output: sampled patch bag with shape [1, K, ...]

import os
import torch
import pandas as pd
from torch.utils.data import Dataset


class PEGTBMILDataset(Dataset):
    # Load slide list and corresponding PT files (binary: Mesenchymal vs Proneural)
    def __init__(self, csv_path, pt_root, K=512):
        df = pd.read_csv(csv_path)

        # Keep only Mesenchymal and Proneural
        df = df[df["GeneExp_Subtype"].isin(["Mesenchymal", "Proneural"])].reset_index(drop=True)

        self.slide_ids = df["sampleID"].values.tolist()
        labels_raw = df["GeneExp_Subtype"].values.tolist()

        # Mapping 2-class subtype → int index
        self.label_map = {
            "Mesenchymal": 0,
            "Proneural": 1,
        }
        self.labels = [self.label_map[x] for x in labels_raw]

        self.pt_root = pt_root
        self.K = K

    # Load slide-level .pt file
    def _load_pt(self, slide_id):
        pt_path = os.path.join(self.pt_root, slide_id + ".pt")
        if not os.path.exists(pt_path):
            return None, None

        data = torch.load(pt_path, weights_only=False)

        # Use the same keys as the original working version
        feat = data["features"].float()      # expected [1, N, 768]
        coord = data["coords"].float()       # expected [1, N, 2]

        if feat.ndim != 3:
            return None, None

        return feat, coord

    # Random sampling of K patches
    def _sample_patches(self, feat, coord):
        N = feat.shape[1]

        if N > self.K:
            idx = torch.randperm(N)[:self.K]
            feat = feat[:, idx, :]
            coord = coord[:, idx, :]
        return feat, coord

    # Prepare one bag for PEGTB model
    def __getitem__(self, idx):
        slide_id = str(self.slide_ids[idx])
        label = int(self.labels[idx])

        feat, coord = self._load_pt(slide_id)
        if feat is None:
            # If missing or broken → return an empty batch (skip)
            return None

        # K=512 sampling
        feat, coord = self._sample_patches(feat, coord)

        return {
            "feat": feat,          # [1, K, 768]
            "coord": coord,        # [1, K, 2]
            "label": label,        # int in {0,1}
            "sample_id": slide_id,
        }

    # Count of slides
    def __len__(self):
        return len(self.slide_ids)


# Custom collate_fn for skipping None samples
def pegtb_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    out = {}
    for key in batch[0].keys():
        if key == "sample_id":
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out
