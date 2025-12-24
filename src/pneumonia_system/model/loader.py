from pathlib import Path
from typing import Tuple, Optional

import torch

from .factory import build_model
from ..mlops.rollback import get_current_version

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def _model_path(version: str) -> Path:
    return MODELS_DIR / version / "model.pth"


def load_model(
    device: torch.device,
    version: Optional[str] = None
) -> Tuple[torch.nn.Module, str]:
    
    if version is None:
        version = get_current_version()

    path = _model_path(version)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    model = build_model(version).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state,strict = True)
    model.eval()

    return model, version
