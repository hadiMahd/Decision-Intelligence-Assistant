from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    answering_model: str = "gpt-4o-mini"

    qdrant_url: str = ""
    qdrant_collection: str = "support_tickets"
    qdrant_vector_size: int = 1536
    qdrant_local_path: str = "artifacts/qdrant"

    external_raw_model_api_url: str = "http://localhost:9000/predict/raw"
    external_engineered_model_api_url: str = "http://localhost:9000/predict/engineered"
    external_model_api_timeout_seconds: int = 15

    top_k: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()


def ensure_artifact_dirs() -> None:
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path(settings.qdrant_local_path).mkdir(parents=True, exist_ok=True)
