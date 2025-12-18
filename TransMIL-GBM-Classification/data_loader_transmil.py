# data_loader_transmil.py (binary classification version)
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class GBMTransMILDataset(Dataset):
    # MIL dataset for TransMIL using CLAM-style feature files (binary version)
    def __init__(self, pt_dir, csv_path):
        super().__init__()
        self.pt_dir = pt_dir

        df = pd.read_csv(csv_path)

        # original 4-class mapping
        subtype_map = {"Classical": 0, "Mesenchymal": 1, "Neural": 2, "Proneural": 3}
        df["label4"] = df["GeneExp_Subtype"].map(subtype_map)

        # filter Mesenchymal(1) and Proneural(3)
        df = df[df["label4"].isin([1, 3])].reset_index(drop=True)

        # convert to binary labels
        # Mesenchymal(1) -> 0
        # Proneural(3)   -> 1
        df["label"] = df["label4"].apply(lambda x: 0 if x == 1 else 1)

        self.sample_ids = df["sampleID"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        label = self.labels[idx]

        # load CLAM-style tensor [N,768]
        pt_path = os.path.join(self.pt_dir, f"{sid}.pt")
        feats = torch.load(pt_path).float()

        # limit patch count
        MAX_PATCH = 400
        if feats.shape[0] > MAX_PATCH:
            sel = torch.randperm(feats.shape[0])[:MAX_PATCH]
            feats = feats[sel]

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
