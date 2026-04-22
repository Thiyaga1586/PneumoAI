from __future__ import annotations

import json

import mlflow
import mlflow.pytorch

from pneumoai.mlops.mlflow_registry import configure_mlflow
from pneumoai.models.loader import load_model_bundle


def main() -> None:
    configure_mlflow()

    model, version, threshold, metadata = load_model_bundle()

    with mlflow.start_run(run_name=f"register-{version}") as run:
        mlflow.log_param("model_version", version)
        mlflow.log_param("threshold", float(threshold))

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"metadata.{key}", value)

        mlflow.pytorch.log_model(model, artifact_path="model")

        print(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "version": version,
                    "threshold": threshold,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()