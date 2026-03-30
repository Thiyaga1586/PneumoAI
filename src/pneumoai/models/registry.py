import json
from pathlib import Path
from pneumoai.common.settings import settings


MODELS_DIR = Path(settings.models_dir)


def get_current_version() -> str:
    registry_path = MODELS_DIR / "registry.json"
    if not registry_path.exists():
        return settings.default_model_version

    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("current", settings.default_model_version)


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