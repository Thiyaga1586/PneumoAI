from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PneumoAI"
    env: str = "dev"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_device: str = "cpu"
    default_model_version: str = "v1"

    models_dir: str = "models"
    runtime_dir: str = "data/runtime"
    sqlite_path: str = "data/runtime/requests.db"

    kafka_enabled: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_predict: str = "pneumoai.predict"

    triton_enabled: bool = False
    triton_url: str = "localhost:8001"

    inference_backend: str = "local"

    triton_fallback_to_local: bool = True
    
    mlflow_tracking_uri: str = "file:./mlruns"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()