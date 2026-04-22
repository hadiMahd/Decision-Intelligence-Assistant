from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    answering_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("ANSWERING_MODEL", "ANSWER_MODEL"),
    )

    qdrant_url: str = ""
    qdrant_host: str = ""
    qdrant_port: int = 6333
    qdrant_collection: str = "support_tickets"
    qdrant_vector_size: int = 1536
    qdrant_local_path: str = "artifacts/qdrant"

    external_raw_model_api_url: str = "http://localhost:9000/predict/raw"
    external_engineered_model_api_url: str = "http://localhost:9000/predict/engineered"
    external_model_api_timeout_seconds: int = 15

    top_k: int = 3

    model_config = SettingsConfigDict(
        env_file=(
            str(Path(__file__).resolve().parents[1] / ".env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def model_post_init(self, __context) -> None:
        if not self.qdrant_url and self.qdrant_host:
            self.qdrant_url = f"http://{self.qdrant_host}:{self.qdrant_port}"


settings = Settings()


def ensure_artifact_dirs() -> None:
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path(settings.qdrant_local_path).mkdir(parents=True, exist_ok=True)
