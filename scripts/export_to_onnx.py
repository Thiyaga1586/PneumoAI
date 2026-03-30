from pathlib import Path

from pneumoai.models.export import export_model_to_onnx


def main():
    version = "v1"
    output_path = Path("model_repository") / "pneumonia_classifier" / "1" / "model.onnx"
    exported = export_model_to_onnx(version=version, output_path=str(output_path))
    print(f"Exported ONNX model to: {exported}")


if __name__ == "__main__":
    main()