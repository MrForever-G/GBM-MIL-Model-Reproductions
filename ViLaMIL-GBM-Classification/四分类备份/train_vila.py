# train_vila.py
# Training loop for ViLa-MIL
# Log format aligned with TwinsGCN / CLAM projects

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluation import full_evaluation, ensemble_patient


def eval_mil(model, loader, device):
    """MIL-style evaluation with patient-level aggregation."""
    model.eval()
    y_true, y_probs, patient_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, ncols=80, desc="Evaluating"):
            patch_feat = batch["patch_feat"].squeeze(0).to(device)
            cluster_feat = batch["cluster_feat"].squeeze(0).to(device)
            label = batch["label"].item()

            sid_raw = batch["sample_id"]
            sid = sid_raw[0] if isinstance(sid_raw, list) else sid_raw

            logits = model(patch_feat, cluster_feat)
            prob = F.softmax(logits, dim=0).cpu().numpy()

            y_true.append(label)
            y_probs.append(prob)
            patient_ids.append(sid)

    import numpy as np
    y_true = np.array(y_true)
    y_probs = np.vstack(y_probs)

    _, ens_probs, ens_true = ensemble_patient(patient_ids, y_true, y_probs)
    auc_low, auc_mid, auc_high = full_evaluation(ens_true, ens_probs)
    return auc_low, auc_mid, auc_high


def train_vila(model, train_loader, test_loader, optimizer,
               num_epochs, device, logger, model_dir):
    """Main training loop for ViLa-MIL."""
    model.to(device)
    best_test_auc = 0.0

    for epoch in range(num_epochs):
        logger.info(f"Epoch[{epoch+1}/{num_epochs}]")
        model.train()
        total_loss, batch_count = 0.0, 0

        # ----- training -----
        for batch in tqdm(train_loader, desc="Loading slides", ncols=80):
            patch_feat = batch["patch_feat"].squeeze(0).to(device)
            cluster_feat = batch["cluster_feat"].squeeze(0).to(device)
            label = batch["label"].to(device)

            logits = model(patch_feat, cluster_feat).unsqueeze(0)
            loss = F.cross_entropy(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(1, batch_count)
        logger.info(
            "Epoch[%d/%d], Total Loss:%.4f (CLF:%.4f)"
            % (epoch + 1, num_epochs, avg_loss, avg_loss)
        )

        # ----- evaluation -----
        train_auc = eval_mil(model, train_loader, device)
        test_auc = eval_mil(model, test_loader, device)

        logger.info(
            "Epoch[%d/%d], train AUC: %.4f(%.4f-%.4f), test AUC: %.4f(%.4f-%.4f)"
            % (
                epoch + 1,
                num_epochs,
                train_auc[1],
                train_auc[0],
                train_auc[2],
                test_auc[1],
                test_auc[0],
                test_auc[2],
            )
        )

        # ----- best model saving (align with TwinsGCN layout) -----
        if test_auc[1] > best_test_auc:
            best_test_auc = test_auc[1]
            logger.info(
                "*** New best model on TEST set! Test AUC: %.4f, saving... ***"
                % best_test_auc
            )
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, "model.pt")
            torch.save(model.state_dict(), save_path)
