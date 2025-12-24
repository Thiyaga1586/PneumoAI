import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..observability.store import recent_requests
from .rollback import rollback


def _find_repo_root(start: Path) -> Path:
    p = start
    for _ in range(10):
        if (p / "models").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find repo root containing /models folder")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
MODELS_DIR = REPO_ROOT / "models"


def psi(expected: List[float], actual: List[float], eps: float = 1e-8) -> float:
    """
    Population Stability Index (PSI): (a-e)*ln(a/e) summed over bins
    """
    e = np.array(expected, dtype=np.float64) + eps
    a = np.array(actual, dtype=np.float64) + eps
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def load_baseline(version: str) -> Tuple[int, List[float]]:
    p = MODELS_DIR / f"baseline_hist_{version}.json"
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return int(obj["bins"]), list(obj["baseline"])


def compute_current_hist(window: int = 200) -> List[float]:
    rows = recent_requests(window)
    hists = []
    for (_ts, _mv, _lat, _label, _prob, hist_json, _err) in rows:
        if hist_json:
            hists.append(np.array(json.loads(hist_json), dtype=np.float64))

    if not hists:
        raise RuntimeError("No histogram data found in recent requests")

    cur = np.mean(np.stack(hists, axis=0), axis=0)
    cur = cur / (cur.sum() + 1e-12)
    return cur.tolist()


def drift_check_and_maybe_rollback(
    version: str,
    window: int = 200,
    threshold: float = 0.25
) -> float:
    _bins, baseline = load_baseline(version)
    current = compute_current_hist(window=window)
    score = psi(baseline, current)

    if score >= threshold:
        rollback()

    return score
