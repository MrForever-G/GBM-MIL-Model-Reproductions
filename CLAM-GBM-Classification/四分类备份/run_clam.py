import os
import json
import datetime
import subprocess

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from utils import check_dir, setup_seed, get_logger
from models_clam import CLAM_MB
from data_loader_clam import CLAMGraphDataset
from torch_geometric.loader import DataLoader
from train_clam import train

import argparse


class savePath:
    def __init__(self, time_tab):
        root = "./TrainProcess/"
        self.model_path = os.path.join(root, time_tab, "model")
        self.writer_path = os.path.join(root, time_tab, "tensorboard", time_tab)
        self.log_path = os.path.join(root, time_tab, f"{time_tab}.log")
        self.argument_path = os.path.join(root, time_tab, f"{time_tab}.json")

        check_dir(self.model_path)
        check_dir(self.writer_path)


def main(args, time_tab):
    save_path = savePath(time_tab)
    logger = get_logger(save_path.log_path)
    writer = SummaryWriter(save_path.writer_path)

    with open(save_path.argument_path, "w") as fw:
        json.dump(args.__dict__, fw, indent=2)

    traindata = CLAMGraphDataset(
        data_dir=args.pt_dir,
        sample_csv=args.train_csv
    )
    testdata = CLAMGraphDataset(
        data_dir=args.pt_dir,
        sample_csv=args.test_csv
    )

    train_loader = DataLoader(traindata, batch_size=1, shuffle=True)
    test_loader = DataLoader(testdata, batch_size=1, shuffle=False)

    model = CLAM_MB(
        in_dim=768,
        hidden_dim=args.hidden_dim,
        n_classes=args.num_classes,
        dropout=args.dropout
    ).to(args.device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = None

    args.logger = logger
    args.writer = writer
    args.model_path = save_path.model_path

    train(model, train_loader, test_loader, loss_func, optimizer, scheduler, args)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--dropout", action="store_true")
    p.add_argument("--savedisk", action="store_true")

    p.add_argument("--pt_dir", type=str, default="./pt_files")
    p.add_argument("--train_csv", type=str, default="./train_sample_id.csv")
    p.add_argument("--test_csv", type=str, default="./test_sample_id.csv")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_seed(2025)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    time_tab = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("Running:", time_tab)

    cmd = f"tensorboard --logdir=./TrainProcess/{time_tab}"
    proc = subprocess.Popen(cmd)

    main(args, time_tab)

    proc.kill()
