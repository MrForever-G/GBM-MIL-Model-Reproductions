import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class GBMTransMILDataset(Dataset):
    # MIL dataset for TransMIL using CLAM-style feature files
    def __init__(self, pt_dir, csv_path):
        super().__init__()
        self.pt_dir = pt_dir

        # CSV contains sampleID and GeneExp_Subtype
        df = pd.read_csv(csv_path)
        self.sample_ids = df["sampleID"].astype(str).tolist()

        # molecular subtype to class index
        subtype_map = {
            "Proneural": 0,
            "Mesenchymal": 1,
            "Classical": 2,
            "Neural": 3,
        }
        self.labels = df["GeneExp_Subtype"].map(subtype_map).tolist()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        label = self.labels[idx]

        # load CLAM-style tensor [N,768]
        pt_path = os.path.join(self.pt_dir, f"{sid}.pt")
        feats = torch.load(pt_path).float()     

        # limit patch count BEFORE coords is created
        MAX_PATCH = 400
        if feats.shape[0] > MAX_PATCH:
            sel = torch.randperm(feats.shape[0])[:MAX_PATCH]
            feats = feats[sel]               # subsampled feats

        # now create coords AFTER feats has final shape
        coords = torch.zeros(feats.shape[0], 2).float()

        return {
            "features": feats,
            "coords": coords,
            "label": torch.tensor(label).long(),
            "sample_id": sid,
        }


def get_loader(pt_dir, csv_path, batch_size=1, num_workers=4, shuffle=False):
    dataset = GBMTransMILDataset(pt_dir, csv_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
