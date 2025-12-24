import json
from pathlib import Path
from typing import Dict, Optional

def _find_repo_root(start: Path) -> Path:
    p = start
    for _ in range(10):
        if (p / "models").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find repo root containing /models folder")

REPO_ROOT = _find_repo_root(Path(__file__).resolve())
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"


def _read_registry() -> Dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"registry.json not found at {REGISTRY_PATH}")
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_registry(reg: Dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)


def get_current_version() -> str:
    reg = _read_registry()
    v = reg.get("current")
    if not v:
        raise ValueError("registry.json missing 'current'")
    return v


def get_previous_version() -> Optional[str]:
    reg = _read_registry()
    return reg.get("previous")


def set_current_version(new_version: str) -> None:
    reg = _read_registry()
    old = reg.get("current")

    reg["previous"] = old
    reg["current"] = new_version

    if "available" not in reg:
        reg["available"] = []
    if new_version not in reg["available"]:
        reg["available"].append(new_version)

    _write_registry(reg)


def rollback() -> str:
    """
    rolls back to previous model version and returns the new current version
    """
    reg = _read_registry()
    prev = reg.get("previous")
    if not prev:
        raise RuntimeError("No previous version available for rollback")

    reg["current"], reg["previous"] = prev, reg.get("current")
    _write_registry(reg)
    return reg["current"]
