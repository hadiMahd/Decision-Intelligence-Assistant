from fastapi import APIRouter, HTTPException

from config import settings
from ml.engineered_feat_model import infer_engineered
from ml.preprocess import preprocess_raw_text
from ml.tf_idf_model import infer_tfidf
from schemas.ml_schemas import MLCompareInferenceRequest, MLCompareInferenceResponse
from services.llm_client import classify_ticket_urgency

router = APIRouter(prefix="/ml", tags=["ml"])


def _normalize_urgency_label(value: str | int | None) -> str:
    if value is None:
        return "not_urgent"

    if isinstance(value, (int, float)):
        return "urgent" if int(value) == 1 else "not_urgent"

    lowered = str(value).strip().lower()
    if lowered in {"1", "urgent", "high", "true", "yes"}:
        return "urgent"
    if lowered in {"0", "not_urgent", "not urgent", "low", "false", "no"}:
        return "not_urgent"

    return "urgent" if "urgent" in lowered and "not" not in lowered else "not_urgent"


@router.post("/compare-inference", response_model=MLCompareInferenceResponse)
def compare_models(payload: MLCompareInferenceRequest) -> MLCompareInferenceResponse:
    cleaned_text, engineered_features = preprocess_raw_text(payload.raw_text)

    try:
        raw_pred, raw_conf = infer_tfidf(raw_text=payload.raw_text, cleaned_text=cleaned_text)
        eng_pred, eng_conf = infer_engineered(engineered_features)
        llm_pred = classify_ticket_urgency(payload.raw_text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Local model artifact missing: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Local model inference failed: {exc}") from exc

    raw_label = _normalize_urgency_label(raw_pred)
    engineered_label = _normalize_urgency_label(eng_pred)

    return MLCompareInferenceResponse(
        raw_model_prediction=raw_pred,
        engineered_model_prediction=eng_pred,
        llm_prediction=llm_pred,
        raw_model_confidence=raw_conf,
        engineered_model_confidence=eng_conf,
        disagreement=raw_pred != eng_pred,
        raw_vs_llm_disagreement=raw_label != llm_pred,
        engineered_vs_llm_disagreement=engineered_label != llm_pred,
        external_raw_response={
            "raw_model": {
                "source": "local_artifact",
                "cleaned_text": cleaned_text,
                "normalized_prediction": raw_label,
            },
            "engineered_model": {
                "source": "local_artifact",
                "feature_order": list(engineered_features.keys()),
                "normalized_prediction": engineered_label,
            },
            "llm_model": {
                "source": "openai" if settings.openai_api_key else "mock",
                "label_space": ["urgent", "not_urgent"],
            },
        },
    )
