from pathlib import Path
import torch

from pneumoai.common.settings import settings
from pneumoai.models.factory import build_model
from pneumoai.models.registry import (
    get_current_version,
    load_threshold,
    load_model_metadata,
)


def _model_path(version: str) -> Path:
    return Path(settings.models_dir) / version / "model.pth"


def load_model_bundle(version: str | None = None):
    if version is None:
        version = get_current_version()

    model = build_model(version)

    model_path = _model_path(version)
    if model_path.exists():
        state = torch.load(model_path, map_location=settings.model_device)
        model.load_state_dict(state, strict=False)

    model.eval()

    threshold = load_threshold(version)
    metadata = load_model_metadata(version)

    return model, version, threshold, metadata