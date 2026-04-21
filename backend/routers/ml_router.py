from fastapi import APIRouter, HTTPException

from ml.preprocess import preprocess_raw_text
from schemas.ml_schemas import MLCompareInferenceRequest, MLCompareInferenceResponse
from services.external_model_client import infer_engineered_model, infer_raw_model

router = APIRouter(prefix="/ml", tags=["ml"])


def _pick(result: dict, keys: list[str]):
    for key in keys:
        if key in result:
            return result[key]
    return None


@router.post("/compare-inference", response_model=MLCompareInferenceResponse)
def compare_models(payload: MLCompareInferenceRequest) -> MLCompareInferenceResponse:
    cleaned_text, engineered_features = preprocess_raw_text(payload.raw_text)

    try:
        raw_model_response = infer_raw_model(
            raw_text=payload.raw_text,
            cleaned_text=cleaned_text,
            engineered_features=engineered_features,
        )
        engineered_model_response = infer_engineered_model(
            raw_text=payload.raw_text,
            cleaned_text=cleaned_text,
            engineered_features=engineered_features,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"External model API failed: {exc}") from exc

    raw_pred = _pick(raw_model_response, ["raw_model_prediction", "raw_prediction", "model1_prediction", "prediction"])
    eng_pred = _pick(
        engineered_model_response,
        ["engineered_model_prediction", "engineered_prediction", "model2_prediction", "prediction"],
    )
    raw_conf = _pick(raw_model_response, ["raw_model_confidence", "raw_confidence", "model1_confidence", "confidence"])
    eng_conf = _pick(
        engineered_model_response,
        ["engineered_model_confidence", "engineered_confidence", "model2_confidence", "confidence"],
    )

    return MLCompareInferenceResponse(
        raw_model_prediction=raw_pred,
        engineered_model_prediction=eng_pred,
        raw_model_confidence=raw_conf,
        engineered_model_confidence=eng_conf,
        disagreement=raw_pred != eng_pred,
        external_raw_response={
            "raw_model": raw_model_response,
            "engineered_model": engineered_model_response,
        },
    )
