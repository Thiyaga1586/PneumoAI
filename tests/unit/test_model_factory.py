from pneumoai.models.factory import build_model
from pneumoai.models.architectures import ImprovedPneumoniaCNN


def test_v2_uses_metadata_driven_baseline_architecture():
    model = build_model("v2")

    assert isinstance(model, ImprovedPneumoniaCNN)