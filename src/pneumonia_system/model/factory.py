import torch.nn as nn
from typing import Dict, Type

from .architecture import (
    ImprovedPneumoniaCNN,
    DeepResNet,
    EfficientNetB0,
)

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "v1": ImprovedPneumoniaCNN,
    "v2": DeepResNet,
    "v3": EfficientNetB0,
}


def build_model(version: str) -> nn.Module:
    if version not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model version: {version}")
    return MODEL_REGISTRY[version]()
