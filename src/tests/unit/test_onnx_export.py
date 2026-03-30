from pathlib import Path

from pneumoai.models.export import export_model_to_onnx


def test_export_model_to_onnx(tmp_path: Path):
    output_path = tmp_path / "model.onnx"

    exported = export_model_to_onnx("v1", str(output_path))

    assert exported is not None
    assert output_path.exists()
    assert output_path.stat().st_size > 0