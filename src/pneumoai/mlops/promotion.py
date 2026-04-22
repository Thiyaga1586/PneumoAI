from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


REGISTRY_PATH = Path("models") / "registry.json"


def promote_model_version(
    version: str,
    run_id: str | None = None,
    notes: str | None = None,
) -> dict:
    registry = {"current": version, "previous": None, "history": []}

    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    previous = registry.get("current")
    registry["previous"] = previous
    registry["current"] = version
    registry.setdefault("history", []).append(
        {
            "version": version,
            "promoted_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "notes": notes,
        }
    )

    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry