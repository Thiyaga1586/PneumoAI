from pneumoai.models.factory import build_model
from pneumoai.models.registry import (
    get_current_version,
    load_threshold,
    load_model_metadata,
)


def load_model_bundle(version: str | None = None):
    if version is None:
        version = get_current_version()

    model = build_model(version)
    threshold = load_threshold(version)
    metadata = load_model_metadata(version)

    return model, version, threshold, metadata