# Exportiert das trainierte PyTorch-Modell in eine .npz Datei fuer numpy_model.py
#
#   python export_numpy.py

import numpy as np
import torch

ckpt = torch.load("checkpoints/ckpt.pt", map_location="cpu")
cfg = ckpt["config"]
weights = {k: v.numpy().astype(np.float16) for k, v in ckpt["model"].items()}
weights["config"] = np.array([cfg["vocab_size"], cfg["block_size"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"]])
np.savez_compressed("checkpoints/model.npz", **weights)
print("gespeichert: checkpoints/model.npz")
