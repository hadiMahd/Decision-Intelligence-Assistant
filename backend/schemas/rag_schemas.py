from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RAGIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rows: int = Field(default=20, ge=1)


class RAGIngestTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    id: str | None = None
    source: str = "manual_test"

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be empty")
        return normalized

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("id must not be empty")
        return normalized

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("source must not be empty")
        return normalized


class RetrievedTicket(BaseModel):
    id: str
    text: str
    source: str
    similarity_score: float | None = None


class RAGCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_text: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("ticket_text")
    @classmethod
    def validate_ticket_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("ticket_text must not be empty")
        return normalized


class RAGCompareResponse(BaseModel):
    no_rag_answer: str
    rag_answer: str
    retrieved_tickets: list[RetrievedTicket | dict[str, Any]]


class RAGSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized


class RAGSearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[RetrievedTicket | dict[str, Any]]
