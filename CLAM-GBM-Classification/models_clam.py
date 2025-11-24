import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn_Net_Gated(nn.Module):
    """
    Attention-gated network used by CLAM.
    Input: [N, D]
    Output: attention scores for N instances
    """
    def __init__(self, in_dim, hidden_dim, dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(in_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(p=0.25) if dropout else None

    def forward(self, x):
        a = torch.tanh(self.fc1(x))
        b = torch.sigmoid(self.fc2(x))

        ab = a * b
        if self.dropout:
            ab = self.dropout(ab)

        A = self.fc3(ab)  # [N, 1]
        return A


class CLAM_MB(nn.Module):
    """
    CLAM-MB: Multi-Branch Attention MIL
    Each class has its own attention branch.
    """
    def __init__(self, in_dim=768, hidden_dim=512, n_classes=4, dropout=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # shared feature encoder
        self.fc1 = nn.Linear(in_dim, hidden_dim)

        # attention branch per class
        self.att_branches = nn.ModuleList([
            Attn_Net_Gated(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(n_classes)
        ])

        # classifier per class
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(n_classes)
        ])

    def forward(self, data):
        # data.x is patch feature [N_patches, 768]
        x = data.x

        H = self.fc1(x)  # [N, hidden_dim]

        bag_logits = []
        att_maps = []

        for k in range(self.n_classes):
            A_k = self.att_branches[k](H)       # [N,1]
            A_k = A_k.transpose(1, 0)           # [1,N]
            A_k = torch.softmax(A_k, dim=1)     # MIL attention

            M_k = torch.mm(A_k, H)              # [1,hidden_dim]
            logit_k = self.classifiers[k](M_k)  # [1,1]

            bag_logits.append(logit_k)
            att_maps.append(A_k)

        logits = torch.cat(bag_logits, dim=1)  # [1,n_classes]
        logits = logits.squeeze(0)             

        return logits
