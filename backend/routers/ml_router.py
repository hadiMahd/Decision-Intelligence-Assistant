from fastapi import APIRouter, HTTPException

from ml.engineered_feat_model import infer_engineered
from ml.preprocess import preprocess_raw_text
from ml.tf_idf_model import infer_tfidf
from schemas.ml_schemas import MLCompareInferenceRequest, MLCompareInferenceResponse

router = APIRouter(prefix="/ml", tags=["ml"])


@router.post("/compare-inference", response_model=MLCompareInferenceResponse)
def compare_models(payload: MLCompareInferenceRequest) -> MLCompareInferenceResponse:
    cleaned_text, engineered_features = preprocess_raw_text(payload.raw_text)

    try:
        raw_pred, raw_conf = infer_tfidf(raw_text=payload.raw_text, cleaned_text=cleaned_text)
        eng_pred, eng_conf = infer_engineered(engineered_features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Local model artifact missing: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Local model inference failed: {exc}") from exc

    return MLCompareInferenceResponse(
        raw_model_prediction=raw_pred,
        engineered_model_prediction=eng_pred,
        raw_model_confidence=raw_conf,
        engineered_model_confidence=eng_conf,
        disagreement=raw_pred != eng_pred,
        external_raw_response={
            "raw_model": {
                "source": "local_artifact",
                "cleaned_text": cleaned_text,
            },
            "engineered_model": {
                "source": "local_artifact",
                "feature_order": list(engineered_features.keys()),
            },
        },
    )
