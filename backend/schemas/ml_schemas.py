from typing import Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MLCompareInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(min_length=1)

    @field_validator("raw_text")
    @classmethod
    def validate_raw_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("raw_text must not be empty")
        return normalized


class MLCompareInferenceResponse(BaseModel):
    raw_model_prediction: str | int | None
    engineered_model_prediction: str | int | None
    llm_prediction: Literal["urgent", "not_urgent"]
    raw_model_confidence: float | None = Field(default=None, ge=0, le=1)
    engineered_model_confidence: float | None = Field(default=None, ge=0, le=1)
    disagreement: bool
    raw_vs_llm_disagreement: bool
    engineered_vs_llm_disagreement: bool
    external_raw_response: dict[str, Any]


class EngineeredFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment_score: float = Field(ge=-1, le=1)
    has_cancel: int = Field(ge=0, le=1)
    has_error: int = Field(ge=0, le=1)
    has_problem: int = Field(ge=0, le=1)
    exclamation_count: int = Field(ge=0)
    has_down: int = Field(ge=0, le=1)
    question_count: int = Field(ge=0)
    has_refund: int = Field(ge=0, le=1)
    has_issue: int = Field(ge=0, le=1)
    has_broken: int = Field(ge=0, le=1)


class EngineeredInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engineered_features: EngineeredFeatures


class EngineeredInferenceResponse(BaseModel):
    prediction: str | int
    confidence: float | None = Field(default=None, ge=0, le=1)


class TfidfInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(min_length=1)

    @field_validator("raw_text")
    @classmethod
    def validate_raw_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("raw_text must not be empty")
        return normalized


class TfidfInferenceResponse(BaseModel):
    prediction: str | int
    confidence: float | None = Field(default=None, ge=0, le=1)
    cleaned_text: str
