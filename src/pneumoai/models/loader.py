from pathlib import Path

import torch

from pneumoai.common.settings import settings
from pneumoai.models.factory import build_model
from pneumoai.models.registry import (
    get_current_version,
    load_model_metadata,
    load_threshold,
)


def resolve_device() -> str:
    if settings.model_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return settings.model_device


def _model_dir(version: str) -> Path:
    return Path(settings.models_dir) / version


def _model_path(version: str) -> Path:
    return _model_dir(version) / "model.pth"


def validate_model_artifacts(version: str) -> None:
    model_dir = _model_dir(version)
    required = [
        model_dir / "model.pth",
        model_dir / "metadata.json",
        model_dir / "threshold.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts for version '{version}': {missing}"
        )


def _extract_state_dict(state: object) -> dict:
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise TypeError("Unsupported model checkpoint format")


def load_model_bundle(version: str | None = None):
    if version is None:
        version = get_current_version()

    validate_model_artifacts(version)

    device = resolve_device()
    model = build_model(version)
    model_path = _model_path(version)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    threshold = load_threshold(version)
    metadata = load_model_metadata(version)

    return model, version, threshold, metadata