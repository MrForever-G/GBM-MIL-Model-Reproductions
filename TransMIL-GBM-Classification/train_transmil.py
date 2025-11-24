import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from models_transmil import TransMIL
from data_loader_transmil import get_loader
from evaluation import ensemble_patient, bootstrap_ap
from utils import get_logger


def train_one_epoch(model, loader, optimizer, device):
    # one epoch of MIL supervised optimization
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for batch in loader:
        feats = batch["features"].to(device).squeeze(0)
        coords = batch["coords"].to(device).squeeze(0)
        label = batch["label"].to(device)

        optimizer.zero_grad()

        logits, _ = model(feats, coords)
        logits = logits.unsqueeze(0)
        loss = criterion(logits, label)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def eval_auc(model, loader, device):
    # compute patient-level AUC with bootstrap (TwinGCN unified style)
    model.eval()

    sample_ids = []
    labels = []
    probs = []

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device).squeeze(0)
            coords = batch["coords"].to(device).squeeze(0)

            logits, _ = model(feats, coords)
            prob = torch.softmax(logits, dim=0).cpu().numpy()

            sid = batch["sample_id"]
            sid = sid[0] if isinstance(sid, (list, tuple)) else sid

            lab = batch["label"]
            lab = lab.item() if isinstance(lab, torch.Tensor) else lab

            sample_ids.append(sid)
            labels.append(lab)
            probs.append(prob)

    import numpy as np
    labels_np = np.array(labels)
    probs_np = np.stack(probs, axis=0)

    _, ens_probs, ens_true = ensemble_patient(sample_ids, labels_np, probs_np)
    low, mid, high = bootstrap_ap(ens_true, ens_probs, B=1000, c=0.95)

    return low, mid, high


def save_pred_csv(save_dir, sample_ids, labels, probs):
    import numpy as np
    import pandas as pd

    probs_np = np.stack(probs, axis=0)
    df = pd.DataFrame({
        "sample_id": sample_ids,
        "label": labels,
        "prob_0": probs_np[:, 0],
        "prob_1": probs_np[:, 1],
        "prob_2": probs_np[:, 2],
        "prob_3": probs_np[:, 3],
    })
    df.to_csv(os.path.join(save_dir, "test_slide_preds.csv"), index=False)


def train_transmil(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)

    # unified logger (TwinGCN style)
    log_path = os.path.join(cfg.save_dir, f"{cfg.time_tag}.log")
    logger = get_logger(log_path)

    logger.info("=== TransMIL Training (Unified GBM-MIL Framework) ===")

    # print HParams (TwinGCN style)
    hparams_str = (
        f"pt_dir={cfg.pt_dir} | train_csv={cfg.train_csv} | test_csv={cfg.test_csv} | "
        f"num_classes={cfg.num_classes} | feat_dim={cfg.feat_dim} | "
        f"epochs={cfg.epochs} | lr={cfg.lr}"
    )
    logger.info(f"HParams: {hparams_str}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # data loader
    train_loader = get_loader(cfg.pt_dir, cfg.train_csv, batch_size=1, num_workers=4, shuffle=True)
    test_loader = get_loader(cfg.pt_dir, cfg.test_csv, batch_size=1, num_workers=4, shuffle=False)

    # build model
    model = TransMIL(n_classes=cfg.num_classes, input_dim=cfg.feat_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    best_mid = -1e9
    best_epoch = 0

    logger.info("Start training")

    for epoch in range(cfg.epochs):
        logger.info(f"Epoch[{epoch+1}/{cfg.epochs}]")
        t0 = time.time()

        # train loss
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        logger.info(f"loss: {train_loss:.4f}")

        # train/test AUC
        low_tr, mid_tr, high_tr = eval_auc(model, train_loader, device)
        low_te, mid_te, high_te = eval_auc(model, test_loader, device)

        logger.info(
            f"train AUC: {mid_tr:.4f}({low_tr:.4f}-{high_tr:.4f}), "
            f"test AUC: {mid_te:.4f}({low_te:.4f}-{high_te:.4f})"
        )

        # save best
        if mid_te > best_mid:
            best_mid = mid_te
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best.pth"))
            logger.info("Updated best model")

        logger.info(f"Epoch time: {time.time()-t0:.2f}s")

    logger.info(f"Finished. Best epoch: {best_epoch}, best AUC(mid): {best_mid:.4f}")
