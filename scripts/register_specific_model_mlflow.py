from __future__ import annotations

import json
import sys

import mlflow
import mlflow.pytorch

from pneumoai.mlops.mlflow_registry import configure_mlflow
from pneumoai.models.loader import load_model_bundle


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/register_specific_model_mlflow.py <version>")

    version = sys.argv[1]

    configure_mlflow()

    model, resolved_version, threshold, metadata = load_model_bundle(version=version)

    with mlflow.start_run(run_name=f"register-{resolved_version}") as run:
        mlflow.log_param("model_version", resolved_version)
        mlflow.log_param("threshold", float(threshold))

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"metadata.{key}", value)

        mlflow.pytorch.log_model(model, name="model")

        print(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "version": resolved_version,
                    "threshold": threshold,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()