import os
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

from ..model.loader import load_model
from ..model.preprocess import val_transform

def evaluate(data_dir: str, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, version, thr = load_model(device=device)
    model.eval()

    ds = datasets.ImageFolder(data_dir, transform=val_transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    print("device:", device)
    print("model_version:", version)
    print("samples:", len(ds))
    print("class_to_idx:", ds.class_to_idx)

    y_true_all = []
    y_prob_all = []

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            y_prob_all.append(probs)
            y_true_all.append(y.numpy())

    y_prob = np.concatenate(y_prob_all)
    y_true = np.concatenate(y_true_all)

    # thr = 0.5
    y_pred = (y_prob >= thr).astype(int)

    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())

    acc = float((y_pred == y_true).mean())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("\n=== Results (thr=0.5) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print("\nConfusion Matrix (NORMAL=0, PNEUMONIA=1)")
    print(f"TN={tn}  FP={fp}")
    print(f"FN={fn}  TP={tp}")

    # extra: average probabilities
    avg_pneu = float(y_prob[y_true == 1].mean()) if (y_true == 1).any() else 0.0
    avg_norm = float(y_prob[y_true == 0].mean()) if (y_true == 0).any() else 0.0
    print(f"\nAvg prob (PNEUMONIA): {avg_pneu:.4f}")
    print(f"Avg prob (NORMAL)   : {avg_norm:.4f}")

if __name__ == "__main__":
    import sys
    evaluate(sys.argv[1], batch_size=32)
