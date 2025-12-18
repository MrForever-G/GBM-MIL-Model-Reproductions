import os
import torch
import pandas as pd
from torch_geometric.data import Dataset, Data


class CLAMGraphDataset(Dataset):
    """
    Binary subtype classification dataset for CLAM.
    Only keep Mesenchymal(1) and Proneural(3).
    Map to: Mesenchymal -> 0, Proneural -> 1
    """

    def __init__(self, data_dir, sample_csv):
        super().__init__()
        self.data_dir = data_dir

        df = pd.read_csv(sample_csv)

        # original label mapping
        label4 = {
            "Classical": 0,
            "Mesenchymal": 1,
            "Neural": 2,
            "Proneural": 3
        }

        # convert original label to one of (0,1,2,3)
        df["label4"] = df["GeneExp_Subtype"].map(label4)

        # filter only 1 / 3  → keep Mesenchymal & Proneural
        df = df[df["label4"].isin([1, 3])].reset_index(drop=True)

        # binary remap
        def map_binary(v):
            if v == 1:
                return 0
            if v == 3:
                return 1
            raise RuntimeError(f"Unexpected label {v}")

        df["label_bin"] = df["label4"].map(map_binary)

        self.sample_ids = df["sampleID"].tolist()
        self.labels = df["label_bin"].tolist()

    def len(self):
        return len(self.sample_ids)

    def get(self, idx):
        sid = self.sample_ids[idx]
        pt_path = os.path.join(self.data_dir, f"{sid}.pt")

        obj = torch.load(pt_path)

        if isinstance(obj, dict) and "feats" in obj:
            feats = obj["feats"]
        elif isinstance(obj, torch.Tensor):
            feats = obj
        else:
            raise ValueError(f"Unrecognized pt: {pt_path}")

        data = Data(
            x=feats.float(),
            y=torch.tensor([self.labels[idx]]).long()
        )
        data.sample_id = sid
        return data
