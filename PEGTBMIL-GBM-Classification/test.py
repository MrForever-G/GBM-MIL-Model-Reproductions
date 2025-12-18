import torch

pt = torch.load(r"F:\ProjectWork\GBM-MIL-Model-Reproductions\PEGTBMIL-GBM-Classification\GBM\patch_pt\TCGA-02-0001-01.pt")

print(pt["features"].shape)
print(pt["coords"].shape)
