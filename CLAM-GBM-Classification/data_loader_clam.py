import torch
from torch_geometric.data import Dataset, Data
import os
import pandas as pd


class CLAMGraphDataset(Dataset):
    """
    Load DINOv2 patch features as MIL bags for CLAM.
    Compatible with:
        1. {"feats": Tensor[N,768]}
        2. Tensor[N,768]
    """
    def __init__(self, data_dir, sample_csv):
        super().__init__()
        self.data_dir = data_dir

        df = pd.read_csv(sample_csv)
        self.sample_ids = df["sampleID"].astype(str).tolist()

        self.label_dict = {
            'Proneural': 0, 'Mesenchymal': 1,
            'Classical': 2, 'Neural': 3
        }
        self.labels = df["GeneExp_Subtype"].map(self.label_dict).tolist()

    def len(self):
        return len(self.sample_ids)

    def get(self, idx):
        sid = self.sample_ids[idx]
        pt_path = os.path.join(self.data_dir, f"{sid}.pt")

        obj = torch.load(pt_path)

        # Case 1: {"feats":Tensor}
        if isinstance(obj, dict) and "feats" in obj:
            feats = obj["feats"]
        # Case 2: Tensor
        elif isinstance(obj, torch.Tensor):
            feats = obj
        else:
            raise ValueError(f"Unrecognized pt format: {pt_path}")

        data = Data(
            x=feats.float(),
            y=torch.tensor([self.labels[idx]]).long(),
        )
        data.sample_id = sid  # PyG Data 

        return data
