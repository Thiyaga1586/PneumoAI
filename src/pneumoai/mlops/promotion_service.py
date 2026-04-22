from __future__ import annotations

import json
from pathlib import Path

from pneumoai.mlops.promotion_gate import should_promote
from pneumoai.models.registry import promote_version


def _load_metrics(path: str) -> dict:
    metrics_path = Path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Metrics file must contain a JSON object: {metrics_path}")

    return data


def promote_with_gate(
    version: str,
    *,
    candidate_metrics_path: str,
    champion_metrics_path: str,
    run_id: str | None = None,
    notes: str | None = None,
    promoted_by: str | None = None,
) -> dict:
    candidate = _load_metrics(candidate_metrics_path)
    champion = _load_metrics(champion_metrics_path)

    approved, reasons = should_promote(candidate, champion)
    if not approved:
        raise ValueError("; ".join(reasons))

    return promote_version(
        version,
        run_id=run_id,
        notes=notes,
        promoted_by=promoted_by,
    )