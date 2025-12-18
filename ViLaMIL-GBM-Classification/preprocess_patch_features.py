# preprocess_patch_features.py
# Merge per-patch .npy features into slide-level .pt files
# Each output pt contains:
#   "features": Tensor [N,768]
#   "coords":   Tensor [N,2]

import os
import numpy as np
import torch
from tqdm import tqdm

# --- Input patch directory (npy per patch) ---
DINOV2_ROOT = r"F:\ProjectWork\GCN\GBM\dinov2_feature"

# --- Output directory inside ViLaMIL folder ---
OUT_ROOT = r"F:\ProjectWork\GBM-MIL-Model-Reproductions\ViLaMIL-GBM-Classification\GBM\patch_pt"

os.makedirs(OUT_ROOT, exist_ok=True)


def load_one_slide(slide_dir):
    """Load all patch npy files under one slide directory."""
    fnames = sorted(os.listdir(slide_dir))
    feats, coords = [], []

    for fname in tqdm(fnames, desc=f"  Loading {os.path.basename(slide_dir)} patches", ncols=80, leave=False):
        if not fname.endswith(".npy"):
            continue

        fpath = os.path.join(slide_dir, fname)
        name = fname[:-4]
        parts = name.split("_")

        # filename must end with _x_y.npy
        if len(parts) < 3:
            continue

        try:
            x = float(parts[-2])
            y = float(parts[-1])
        except:
            continue

        try:
            arr = np.load(fpath)
        except Exception as e:
            print(f"[WARNING] Failed to load: {fpath} ({e})")
            continue

        if arr.ndim != 1:
            print(f"[WARNING] Invalid feature shape in {fpath}, skip.")
            continue

        feats.append(arr.astype(np.float32))
        coords.append([x, y])

    if len(feats) == 0:
        return None, None

    feats = np.stack(feats, axis=0)          # [N, 768]
    coords = np.array(coords, dtype=np.float32)  # [N, 2]
    return feats, coords


def main():
    # Check whether dinov2 directory exists
    if not os.path.exists(DINOV2_ROOT):
        raise FileNotFoundError(f"Input folder not found: {DINOV2_ROOT}")

    slide_list = sorted([
        d for d in os.listdir(DINOV2_ROOT)
        if os.path.isdir(os.path.join(DINOV2_ROOT, d))
    ])

    if len(slide_list) == 0:
        raise RuntimeError("No slide directories found in dinov2_feature. Check your path!")

    print(f"Found {len(slide_list)} slides. Starting preprocessing...")

    for slide_id in tqdm(slide_list, desc="Processing slides", ncols=100):
        slide_dir = os.path.join(DINOV2_ROOT, slide_id)
        out_path = os.path.join(OUT_ROOT, f"{slide_id}.pt")

        # skip if already processed
        if os.path.exists(out_path):
            continue

        feats, coords = load_one_slide(slide_dir)
        if feats is None:
            print(f"[WARNING] Slide {slide_id}: no valid patches found. Skip.")
            continue

        torch.save({
            "features": torch.from_numpy(feats),
            "coords": torch.from_numpy(coords),
        }, out_path)

    print("\nAll slide-level .pt files saved to:")
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
