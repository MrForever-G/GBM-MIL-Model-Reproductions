# convert_pt_for_pegtb.py
# Convert ViLa patch_pt format to PEGTB-MIL format.

import os
import torch
from tqdm import tqdm

# Path to ViLa patch_pt
VILA_PT_ROOT = r"F:\ProjectWork\GBM-MIL-Model-Reproductions\ViLaMIL-GBM-Classification\GBM\patch_pt"

# Output path for PEGTB-MIL
PEGTB_PT_ROOT = r"F:\ProjectWork\GBM-MIL-Model-Reproductions\PEGTBMIL-GBM-Classification\GBM\patch_pt"
os.makedirs(PEGTB_PT_ROOT, exist_ok=True)

all_files = [f for f in os.listdir(VILA_PT_ROOT) if f.endswith(".pt")]
print(f"Found {len(all_files)} pt files.")

for fname in tqdm(all_files, desc="Converting pt files"):
    src = os.path.join(VILA_PT_ROOT, fname)
    dst = os.path.join(PEGTB_PT_ROOT, fname)

    data = torch.load(src)

    feat = data["features"]      # [N, 768]
    coord = data["coords"]       # [N, 2]

    # Add batch dimension: [1, N, D]
    feat = feat.unsqueeze(0)     # [1, N, 768]
    coord = coord.unsqueeze(0)   # [1, N, 2]

    new_data = {
        "features": feat,
        "coords": coord
    }

    torch.save(new_data, dst)

print("All pt files converted successfully.")
