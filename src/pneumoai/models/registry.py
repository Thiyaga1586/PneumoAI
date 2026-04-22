from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pneumoai.common.settings import settings

MODELS_DIR = Path(settings.models_dir)
REGISTRY_PATH = MODELS_DIR / "registry.json"


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {
            "current": settings.default_model_version,
            "previous": None,
            "available": [settings.default_model_version],
            "history": [],
        }

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.setdefault("current", settings.default_model_version)
    data.setdefault("previous", None)
    data.setdefault("available", [data["current"]])
    data.setdefault("history", [])

    if data["current"] not in data["available"]:
        data["available"].append(data["current"])

    return data


def _save_registry(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_registry() -> dict:
    return _load_registry()


def get_current_version() -> str:
    return _load_registry()["current"]


def load_threshold(version: str) -> float:
    threshold_path = MODELS_DIR / version / "threshold.json"
    if not threshold_path.exists():
        return 0.5

    with open(threshold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return float(data.get("threshold", 0.5))


def load_model_metadata(version: str) -> dict:
    metadata_path = MODELS_DIR / version / "metadata.json"
    if not metadata_path.exists():
        return {"version": version}

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_promotable_version(version: str) -> None:
    version_dir = MODELS_DIR / version
    required = [
        version_dir / "model.pth",
        version_dir / "metadata.json",
        version_dir / "threshold.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot promote version '{version}'. Missing: {missing}")


def promote_version(
    version: str,
    *,
    run_id: str | None = None,
    notes: str | None = None,
    promoted_by: str | None = None,
) -> dict:
    _validate_promotable_version(version)

    registry = _load_registry()
    current = registry["current"]

    registry["previous"] = current
    registry["current"] = version

    if version not in registry["available"]:
        registry["available"].append(version)

    registry.setdefault("history", []).append(
        {
            "event": "promote",
            "version": version,
            "previous": current,
            "run_id": run_id,
            "notes": notes,
            "promoted_by": promoted_by,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )

    _save_registry(registry)
    return registry


def rollback_version(
    *,
    notes: str | None = None,
    rolled_back_by: str | None = None,
) -> dict:
    registry = _load_registry()
    previous = registry.get("previous")

    if not previous:
        raise ValueError("No previous version available for rollback")

    current = registry["current"]
    registry["current"] = previous
    registry["previous"] = current

    if previous not in registry["available"]:
        registry["available"].append(previous)
    if current not in registry["available"]:
        registry["available"].append(current)

    registry.setdefault("history", []).append(
        {
            "event": "rollback",
            "version": previous,
            "previous": current,
            "notes": notes,
            "rolled_back_by": rolled_back_by,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )

    _save_registry(registry)
    return registry