# train_pegtb.py
# Training loop for PEGTB-MIL with TwinsGCN-aligned logging.

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluation import full_evaluation, ensemble_patient


def eval_mil(model, loader, device):
    # Patient-level MIL evaluation
    model.eval()
    y_true, y_probs, patient_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, ncols=80, desc="Evaluating"):
            if batch is None:
                continue

            feat = batch["feat"].to(device).squeeze(1)      # [B, K, 768]
            coord = batch["coord"].to(device).squeeze(1)    # [B, K, 2]

            # label 是 Python int → 转 tensor
            label = torch.tensor(batch["label"], dtype=torch.long).cpu().item()
            sid = batch["sample_id"][0] if isinstance(batch["sample_id"], list) else batch["sample_id"]

            logits = model(feat, coord).squeeze(0)
            prob = F.softmax(logits, dim=0).cpu().numpy()

            y_true.append(label)
            y_probs.append(prob)
            patient_ids.append(sid)

    import numpy as np
    y_true = np.array(y_true)
    y_probs = np.vstack(y_probs)

    _, ens_probs, ens_true = ensemble_patient(patient_ids, y_true, y_probs)
    return full_evaluation(ens_true, ens_probs)


def train_pegtb(model, train_loader, test_loader,
                optimizer, num_epochs, device, logger, model_dir):

    model.to(device)
    best_test_auc = 0.0

    for epoch in range(num_epochs):
        logger.info(f"Epoch[{epoch+1}/{num_epochs}]")
        model.train()

        total_loss, count = 0.0, 0
        
        for batch in tqdm(train_loader, desc="Loading slides", ncols=80):
            if batch is None:
                continue

            feat = batch["feat"].to(device).squeeze(1)      # [B, K, 768]
            coord = batch["coord"].to(device).squeeze(1)    # [B, K, 2]

            # label → tensor
            label = torch.tensor(batch["label"], dtype=torch.long, device=device)

            logits = model(feat, coord)    # [B, C]
            loss = F.cross_entropy(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(1, count)
        logger.info(f"Epoch[{epoch+1}/{num_epochs}], Total Loss:{avg_loss:.4f} (CLF:{avg_loss:.4f})")

        train_auc = eval_mil(model, train_loader, device)
        test_auc = eval_mil(model, test_loader, device)

        logger.info(
            "Epoch[%d/%d], train AUC: %.4f(%.4f-%.4f), test AUC: %.4f(%.4f-%.4f)"
            % (
                epoch + 1, num_epochs,
                train_auc[1], train_auc[0], train_auc[2],
                test_auc[1], test_auc[0], test_auc[2],
            )
        )

        if test_auc[1] > best_test_auc:
            best_test_auc = test_auc[1]
            logger.info(f"*** New best model on TEST set! Test AUC: {best_test_auc:.4f}, saving... ***")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
