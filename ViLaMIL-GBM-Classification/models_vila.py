# models_vila.py
# Dual-scale Vision-Language MIL for GBM WSI classification
# Patch-level and cluster-level features are fused with text prompts via cross-attention

import torch
import torch.nn as nn
import open_clip


class PatchProjector(nn.Module):
    # Linear projection for patch-level features
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.fc(x))


class ClusterProjector(nn.Module):
    # Linear projection for cluster-level (region-level) features
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.fc(x))


class TextProjector(nn.Module):
    # Projection for CLIP text embeddings
    def __init__(self, in_dim=512, out_dim=256):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.fc(x))


class MultiheadCrossAttention(nn.Module):
    # Cross-attention between vision tokens and text prompts
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, vision_tokens, text_tokens):
        # text_tokens: [1, C, E] act as queries
        # vision_tokens: [1, N, E] act as keys/values
        out, _ = self.attn(query=text_tokens, key=vision_tokens, value=vision_tokens)
        return out


class ViLaMIL(nn.Module):
    # Main ViLa-MIL model
    def __init__(self, num_classes=2, embed_dim=256, num_heads=4):
        super().__init__()

        # Projection layers
        self.patch_proj = PatchProjector(768, embed_dim)
        self.cluster_proj = ClusterProjector(768, embed_dim)
        self.text_proj = TextProjector(512, embed_dim)

        # Pre-compute CLIP text embeddings and store as buffer
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        prompts = [
            "a histopathology image of glioblastoma",
            "a histopathology image of astrocytoma",
            "a histopathology image of oligodendroglioma",
            "a histopathology image of other tumor",
        ]
        with torch.no_grad():
            text_tokens = tokenizer(prompts)
            text_emb = model.encode_text(text_tokens)  # [C, 512]

        self.register_buffer("text_emb", text_emb)

        # Cross-attention modules for small and large scale
        self.attn_small = MultiheadCrossAttention(embed_dim, num_heads)
        self.attn_large = MultiheadCrossAttention(embed_dim, num_heads)

        # Final classifier
        self.fc = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, patch_feat, cluster_feat):
        # patch_feat:   [N, 768]
        # cluster_feat: [K, 768]

        device = patch_feat.device
        text_emb = self.text_emb.to(device)          # [C, 512]
        text_tokens = self.text_proj(text_emb)       # [C, E]

        patch_emb = self.patch_proj(patch_feat)      # [N, E]
        cluster_emb = self.cluster_proj(cluster_feat)  # [K, E]

        # Add batch dimension
        patch_emb = patch_emb.unsqueeze(0)           # [1, N, E]
        cluster_emb = cluster_emb.unsqueeze(0)       # [1, K, E]
        text_tokens = text_tokens.unsqueeze(0)       # [1, C, E]

        # Cross-attention at two scales
        small_att = self.attn_small(patch_emb, text_tokens)    # [1, C, E]
        large_att = self.attn_large(cluster_emb, text_tokens)  # [1, C, E]

        # Class-wise pooling and scale fusion
        small_vec = small_att.mean(dim=1)            # [1, E]
        large_vec = large_att.mean(dim=1)            # [1, E]
        bag_vec = torch.cat([small_vec, large_vec], dim=-1)  # [1, 2E]

        logits = self.fc(bag_vec.squeeze(0))         # [C]
        return logits
