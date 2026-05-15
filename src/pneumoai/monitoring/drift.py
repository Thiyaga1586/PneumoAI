from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import text

from pneumoai.common.settings import settings
from pneumoai.storage.sqlite import get_engine, init_db


def _baseline_hist_path(version: str) -> Path:
    return Path(settings.models_dir) / version / f"baseline_hist_{version}.json"


def load_baseline_histogram(version: str) -> dict[str, Any]:
    path = _baseline_hist_path(version)
    if not path.exists():
        raise FileNotFoundError(f"Baseline histogram file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    bins = int(data["bins"])
    baseline = data["baseline"]
    hist_range = data.get("range", [0.0, 1.0])

    if len(baseline) != bins:
        raise ValueError(
            f"Baseline histogram length {len(baseline)} does not match bins={bins}"
        )

    baseline_arr = np.asarray(baseline, dtype=np.float64)
    total = baseline_arr.sum()
    if total <= 0:
        raise ValueError("Baseline histogram sum must be > 0")

    baseline_arr = baseline_arr / total

    return {
        "version": data.get("version", version),
        "bins": bins,
        "range": hist_range,
        "baseline": baseline_arr,
    }


def fetch_recent_probabilities(
    model_version: str,
    limit: int = 500,
) -> list[float]:
    init_db()

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT probability
                FROM predictions
                WHERE model_version = :model_version
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {
                "model_version": model_version,
                "limit": int(limit),
            },
        ).fetchall()

    probs: list[float] = []
    for row in rows:
        value = row[0]
        if value is not None:
            probs.append(float(value))

    return probs


def compute_live_histogram(
    probabilities: list[float],
    bins: int,
    hist_range: list[float],
) -> np.ndarray:
    if not probabilities:
        raise ValueError("No live probabilities available to compute histogram")

    arr = np.asarray(probabilities, dtype=np.float64)
    arr = np.clip(arr, hist_range[0], hist_range[1])

    hist, _ = np.histogram(arr, bins=bins, range=tuple(hist_range), density=False)
    hist = hist.astype(np.float64)

    total = hist.sum()
    if total <= 0:
        raise ValueError("Computed live histogram has zero total count")

    return hist / total


def _safe_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-12,
) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    jsd = 0.5 * _safe_kl_divergence(p, m, eps) + 0.5 * _safe_kl_divergence(q, m, eps)
    return float(jsd)


def detect_prediction_drift(
    version: str,
    limit: int = 500,
    threshold: float = 0.08,
) -> dict[str, Any]:
    baseline_info = load_baseline_histogram(version)
    live_probabilities = fetch_recent_probabilities(version, limit=limit)

    if len(live_probabilities) < 20:
        return {
            "version": version,
            "drift_detected": False,
            "reason": "insufficient_live_samples",
            "sample_count": len(live_probabilities),
            "threshold": threshold,
            "js_divergence": None,
            "baseline_histogram": baseline_info["baseline"].tolist(),
            "live_histogram": None,
        }

    live_hist = compute_live_histogram(
        probabilities=live_probabilities,
        bins=baseline_info["bins"],
        hist_range=baseline_info["range"],
    )

    jsd = jensen_shannon_divergence(baseline_info["baseline"], live_hist)

    return {
        "version": version,
        "drift_detected": jsd > threshold,
        "reason": "ok",
        "sample_count": len(live_probabilities),
        "threshold": threshold,
        "js_divergence": jsd,
        "baseline_histogram": baseline_info["baseline"].tolist(),
        "live_histogram": live_hist.tolist(),
    }