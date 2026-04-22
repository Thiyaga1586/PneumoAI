from __future__ import annotations

from typing import Any

import mlflow

from pneumoai.common.settings import settings


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def start_run(run_name: str | None = None):
    configure_mlflow()
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict[str, Any]) -> None:
    if params:
        mlflow.log_params(params)


def log_metrics(metrics: dict[str, float]) -> None:
    if metrics:
        mlflow.log_metrics(metrics)


def set_tags(tags: dict[str, str]) -> None:
    if tags:
        mlflow.set_tags(tags)


def get_active_run_id() -> str | None:
    run = mlflow.active_run()
    return run.info.run_id if run else None