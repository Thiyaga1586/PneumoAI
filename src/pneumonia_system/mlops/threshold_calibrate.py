import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

from tqdm import tqdm  

from ..model.loader import load_model
from ..model.preprocess import val_transform


def best_threshold(probs, y_true):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (probs >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def run(val_dir: str, version: str = "v1", batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")  # ✅ quick visibility

    model, loaded_version = load_model(device=device)
    assert loaded_version == version, f"registry is {loaded_version}, expected {version}"

    ds = datasets.ImageFolder(val_dir, transform=val_transform)

    # ✅ speed + avoid “stuck” feeling
    num_workers = 4  # try 2/4/8 depending on your CPU
    pin_memory = (device.type == "cuda")

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    all_probs = []
    all_y = []

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dl, desc="Calibrating threshold", unit="batch"):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.numpy())

    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_y)

    t, f1 = best_threshold(probs, y_true)

    out = Path("models") / version / "threshold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"version": version, "threshold": t, "metric": "f1", "f1": f1}, indent=2))
    print("Saved:", out, "threshold=", t, "f1=", f1)


if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "v1")
