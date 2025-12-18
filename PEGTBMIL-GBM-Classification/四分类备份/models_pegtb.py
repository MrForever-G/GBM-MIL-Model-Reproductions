# models_pegtb.py
# PEGTB-MIL model using coordinate-guided positional encoding and a Transformer encoder.

import torch
import torch.nn as nn


class PatchProjection(nn.Module):
    # Patch feature projection: [1, N, 768] → [1, N, 256]
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class CoordEncoder(nn.Module):
    # Coordinate projection: [1, N, 2] → [1, N, 256]
    def __init__(self, in_dim=2, out_dim=256):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class PEGTBMIL(nn.Module):
    # PEGTB-MIL model with positional encoding-guided transformer MIL
    def __init__(self, num_classes=4, embed_dim=256, num_heads=8, depth=2):
        super().__init__()

        self.patch_proj = PatchProjection(768, embed_dim)
        self.coord_proj = CoordEncoder(2, embed_dim)

        # Learnable PGT token
        self.pgt = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, feat, coord):
        # feat:   [1, N, 768]
        # coord:  [1, N, 2]

        tokens = self.patch_proj(feat) + self.coord_proj(coord)   # [1, N, 256]

        pgt = self.pgt.expand(1, -1, -1)                          # [1, 1, 256]
        tokens = torch.cat([pgt, tokens], dim=1)                  # [1, N+1, 256]

        encoded = self.encoder(tokens)                            # [1, N+1, 256]
        bag_repr = encoded[:, 0, :]                               # [1, 256]

        logits = self.classifier(bag_repr).squeeze(0)             # [C]
        return logits
