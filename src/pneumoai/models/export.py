from pathlib import Path

import torch

from pneumoai.models.loader import load_model_bundle


def export_model_to_onnx(version: str, output_path: str) -> str:
    model, loaded_version, threshold, metadata = load_model_bundle(version)

    model = model.to("cpu")
    model.eval()

    dummy = torch.randn(1, 1, 224, 224, dtype=torch.float32, device="cpu")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        str(output_file),
        input_names=["input_image"],
        output_names=["logits"],
        opset_version=18,
    )

    return str(output_file)