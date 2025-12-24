import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from torchvision import datasets

from ..model.preprocess import resize_with_padding, extract_intensity_histogram


def _find_repo_root(start: Path) -> Path:
    p = start
    for _ in range(10):
        if (p / "models").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find repo root containing /models folder")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
MODELS_DIR = REPO_ROOT / "models"


def compute_baseline_hist(train_dir: str, bins: int = 32, max_images: int = 2000) -> List[float]:
    """
    Computes mean intensity histogram over training images.
    Uses deterministic resize_with_padding + grayscale.
    """
    ds = datasets.ImageFolder(train_dir)
    hists = []
    n = 0

    for path, _ in ds.samples:
        img = Image.open(path)
        img = resize_with_padding(img, 224)  # grayscale PIL 224x224
        gray_np = np.array(img, dtype=np.uint8)
        h = np.array(extract_intensity_histogram(gray_np, bins=bins), dtype=np.float64)
        hists.append(h)
        n += 1
        if n >= max_images:
            break

    if not hists:
        raise RuntimeError("No images found to compute baseline")

    baseline = np.mean(np.stack(hists, axis=0), axis=0)
    baseline = baseline / (baseline.sum() + 1e-12)
    return baseline.tolist()


def save_baseline(version: str, train_dir: str, bins: int = 32, max_images: int = 2000) -> Path:
    baseline = compute_baseline_hist(train_dir=train_dir, bins=bins, max_images=max_images)
    out_path = MODELS_DIR / f"baseline_hist_{version}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"version": version, "bins": bins, "baseline": baseline}, f, indent=2)
    return out_path
