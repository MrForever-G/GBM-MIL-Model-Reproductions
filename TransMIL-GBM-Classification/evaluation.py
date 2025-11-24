import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def save_record(patchID, y_true, y_probs):
    record_dict = dict()
    record_dict["patchID"] = patchID
    record_dict["y_true"] = y_true
    for i in range(y_probs.shape[1]):
        record_dict[f"y_probs{i}"] = y_probs[:, i]
    import pandas as pd
    record_df = pd.DataFrame.from_dict(record_dict)
    return record_df


def ensemble_patient(patientID, y_true, y_probs):
    # ensure patientID list contains pure strings
    clean_ids = []
    for pid in patientID:
        if isinstance(pid, (list, tuple)):
            pid = pid[0]
        clean_ids.append(pid)

    uq_ids = list(set(clean_ids))

    ensemble_y_probs = []
    ensemble_y_true = []

    for uid in uq_ids:
        idx = [uid == pid for pid in clean_ids]
        ensemble_y_probs.append(y_probs[idx, :].mean(axis=0))
        ensemble_y_true.append(y_true[idx].mean())

    ensemble_y_probs = np.array(ensemble_y_probs)
    ensemble_y_true = np.array(ensemble_y_true).ravel()

    return uq_ids, ensemble_y_probs, ensemble_y_true


def auc(target, score):
    if len(set(target)) > 2:
        return roc_auc_score(target, score, average="macro", multi_class="ovr")
    return roc_auc_score(target, score[:, 1])


def bootstrap_ap(target, score, B=1000, c=0.95):
    n = len(target)
    auc_list = []
    count = 0
    while True:
        idx = np.random.randint(0, n, size=n)
        t_sample = target[idx]
        if len(set(t_sample)) == 1:
            continue
        s_sample = score[idx]
        auc_list.append(auc(t_sample, s_sample))
        count += 1
        if count >= B:
            break

    a = 1 - c
    k1 = int(count * a / 2)
    k2 = int(count * (1 - a / 2))
    auc_sorted = sorted(auc_list)
    return auc_sorted[k1], auc_sorted[count // 2], auc_sorted[k2]


def eval_model(model, loader, device):
    y_true = []
    y_probs = []
    patientID = []

    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # normalize patient ID (PyG may wrap it as list)
            pid = data.sample_id
            if isinstance(pid, (list, tuple)):
                pid = pid[0]
            patientID.append(pid)

            logits = model(data)

            # CLAM output is [C], expand to [1,C]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            prob = torch.softmax(logits, dim=1)

            y_probs.append(prob)
            y_true.append(data.y)

    y_true = torch.cat(y_true, dim=0).cpu().numpy().ravel()
    y_probs = torch.cat(y_probs, dim=0).cpu().numpy()

    _, ensemble_y_probs, ensemble_y_true = ensemble_patient(patientID, y_true, y_probs)

    return bootstrap_ap(ensemble_y_true, ensemble_y_probs, 1000, 0.95)
