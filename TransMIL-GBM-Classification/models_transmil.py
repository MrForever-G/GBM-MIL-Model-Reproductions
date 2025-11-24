import torch
import torch.nn as nn
import torch.nn.functional as F


# ...
class PPEG(nn.Module):
    # Position Encoding via Gaussian Kernel (official TransMIL)
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, coords):
        # x: [1, C, H, W] after reshaping
        # coords only determine shape, grid not required
        return x + self.proj(x)


# ...
class TransMIL(nn.Module):
    # Official TransMIL MIL Transformer
    def __init__(self, n_classes=4, input_dim=768, dim=256, depth=4, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()

        # patch embedding
        self.project = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

        # class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))

        # transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # PPEG module
        self.ppeg = PPEG(dim)

        # classifier for MIL prediction
        self.classifier = nn.Linear(dim, n_classes)

        # init
        nn.init.trunc_normal_(self.class_token, std=0.02)

    # ...
    def forward(self, feats, coords):
        # feats: [N, 768]
        # coords: [N, 2]

        N = feats.size(0)

        # patch embedding
        x = self.project(feats)                  # [N, dim]

        # append class token
        cls_tok = self.class_token.expand(1, -1, -1)  # [1, 1, dim]
        x = torch.cat([cls_tok.squeeze(0), x], dim=0)  # [N+1, dim]
        x = x.unsqueeze(0)                            # [1, N+1, dim]

        # prepare 2D PPEG grid
        # simple heuristic grid: derive approximate H, W from N
        H = int(torch.sqrt(torch.tensor(N)).item())
        W = H
        if H * W < N:
            W += 1
        if H * W < N:
            H += 1

        # pad tokens (excluding cls_tok)
        pad_len = H * W - N
        if pad_len > 0:
            pad_tokens = torch.zeros(pad_len, x.size(-1), device=x.device)
            x_ = torch.cat([x[:, 1:], pad_tokens.unsqueeze(0)], dim=1)  # [1, H*W, dim]
        else:
            x_ = x[:, 1:]  # [1, N, dim]

        # reshape into 2D grid
        x_2d = x_.permute(0, 2, 1).reshape(1, x.size(-1), H, W)  # [1, dim, H, W]

        # PPEG
        x_2d = self.ppeg(x_2d, coords)

        # flatten back
        x_ppeg = x_2d.reshape(1, x.size(-1), -1).permute(0, 2, 1)  # [1, H*W, dim]
        x_ppeg = x_ppeg[:, :N, :]  # keep valid tokens
        x = torch.cat([x[:, 0:1, :], x_ppeg], dim=1)  # restore cls_token

        # transformer encoding
        attn_map = []
        for layer in self.encoder.layers:
            x_temp = layer.self_attn(
                x, x, x,
                need_weights=True,
                attn_mask=None
            )
            attn_map.append(x_temp[1])  # attention weights
            x = layer(x)

        # cls_token for classification
        cls_feat = x[:, 0, :]                    # [1, dim]
        logits = self.classifier(cls_feat)       # [1, n_classes]

        # return logits & attention maps
        return logits.squeeze(0), attn_map
