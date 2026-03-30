from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "PneumoAI"
    env: str = "dev"
    model_device: str = "cpu"
    sqlite_path: str = "data/runtime/requests.db"
    
    # MLflow & Triton placeholders for Phase 3
    triton_enabled: bool = False
    mlflow_tracking_uri: str = "file:./mlruns"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()