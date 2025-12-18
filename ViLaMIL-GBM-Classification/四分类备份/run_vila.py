# run_vila.py
# Main entry for ViLa-MIL GBM classification
# Path layout and logging style follow the main GCN project

import os
import argparse
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from data_loader_vila import ViLaMILDataset
from models_vila import ViLaMIL
from train_vila import train_vila
from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run ViLa-MIL GBM Classification")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"Running time tab:{time_stamp}")

    # ----- paths -----
    DATA_ROOT = r"F:\ProjectWork\GCN\GBM"
    PT_ROOT = r"F:\ProjectWork\GBM-MIL-Model-Reproductions\ViLaMIL-GBM-Classification\GBM\patch_pt"
    CLUSTER_ROOT = os.path.join(DATA_ROOT, "postdata", "adaptive_kmeans")

    TRAIN_CSV = os.path.join(DATA_ROOT, "train_sample_id.csv")
    TEST_CSV = os.path.join(DATA_ROOT, "test_sample_id.csv")

    # ----- experiment directory (align with TwinsGCN/CLAM style) -----
    base_dir = "TrainProcess"
    save_dir = os.path.join(base_dir, time_stamp)
    os.makedirs(save_dir, exist_ok=True)

    model_dir = os.path.join(save_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    log_path = os.path.join(save_dir, f"{time_stamp}.log")
    logger = get_logger(log_path)

    logger.info(
        "LEARNING RATE: %.6f, NUM EPOCHS: %d, BATCH SIZE: %d, DEVICE: %s"
        % (args.lr, args.num_epochs, args.batch_size, args.device)
    )

    # ----- dataset -----
    train_dataset = ViLaMILDataset(TRAIN_CSV, PT_ROOT, CLUSTER_ROOT)
    test_dataset = ViLaMILDataset(TEST_CSV, PT_ROOT, CLUSTER_ROOT)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ----- model & optimizer -----
    model = ViLaMIL().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ----- training -----
    train_vila(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        logger=logger,
        model_dir=model_dir,    
    )


if __name__ == "__main__":
    main()
