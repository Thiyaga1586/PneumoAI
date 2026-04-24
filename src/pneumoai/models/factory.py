from __future__ import annotations

from typing import Dict, Type
import json
from pathlib import Path

import torch.nn as nn

from pneumoai.common.settings import settings
from pneumoai.models.architectures import (
    ImprovedPneumoniaCNN,
    DeepResNet,
    EfficientNetB0,
)


ARCH_REGISTRY: Dict[str, Type[nn.Module]] = {
    "baseline": ImprovedPneumoniaCNN,
    "deep_resnet": DeepResNet,
    "efficientnet_b0": EfficientNetB0,
}


def _metadata_path(version: str) -> Path:
    return Path(settings.models_dir) / version / "metadata.json"


def _load_architecture_name(version: str) -> str:
    path = _metadata_path(version)
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata.json for version '{version}'")

    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    architecture = metadata.get("architecture")
    if not architecture:
        raise ValueError(f"Missing 'architecture' field in metadata for version '{version}'")

    return architecture


def build_model(version: str) -> nn.Module:
    architecture = _load_architecture_name(version)

    if architecture not in ARCH_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}' for version '{version}'. "
            f"Supported: {list(ARCH_REGISTRY.keys())}"
        )

    return ARCH_REGISTRY[architecture]()