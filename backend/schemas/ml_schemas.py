from pydantic import BaseModel, Field


class MLCompareInferenceRequest(BaseModel):
    raw_text: str = Field(min_length=1)


class MLCompareInferenceResponse(BaseModel):
    raw_model_prediction: str | int | None
    engineered_model_prediction: str | int | None
    raw_model_confidence: float | None
    engineered_model_confidence: float | None
    disagreement: bool
    external_raw_response: dict


class EngineeredFeatures(BaseModel):
    sentiment_score: float
    has_cancel: int
    has_error: int
    has_problem: int
    exclamation_count: int
    has_down: int
    question_count: int
    has_refund: int
    has_issue: int
    has_broken: int


class EngineeredInferenceRequest(BaseModel):
    engineered_features: EngineeredFeatures


class EngineeredInferenceResponse(BaseModel):
    prediction: str | int
    confidence: float | None


class TfidfInferenceRequest(BaseModel):
    raw_text: str = Field(min_length=1)


class TfidfInferenceResponse(BaseModel):
    prediction: str | int
    confidence: float | None
    cleaned_text: str
