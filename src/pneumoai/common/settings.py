from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PneumoAI"
    env: str = "dev"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    log_level: str = "INFO"

    model_device: str = "auto"
    default_model_version: str = "v1"

    models_dir: str = "models"
    runtime_dir: str = "data/runtime"
    sqlite_path: str = "data/runtime/requests.db"

    inference_backend: str = "local"
    triton_enabled: bool = False
    triton_url: str = "localhost:8001"
    triton_fallback_to_local: bool = True

    max_upload_mb: int = 10
    admin_api_key: str = "change-me"

    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "pneumoai"

    redis_enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    redis_queue_key: str = "pneumoai:queue:predict"
    worker_poll_interval_seconds: float = 1.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()