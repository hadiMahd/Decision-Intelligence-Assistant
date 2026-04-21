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
