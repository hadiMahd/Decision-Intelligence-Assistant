import requests
import pandas as pd

from config import settings


def _base_payload(raw_text: str, cleaned_text: str, engineered_features: dict) -> dict:
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "engineered_features": engineered_features,
        "feature_order": list(engineered_features.keys()),
    }


def infer_raw_model(raw_text: str, cleaned_text: str, engineered_features: dict) -> dict:
    payload = _base_payload(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        engineered_features=engineered_features,
    )
    response = requests.post(
        settings.external_raw_model_api_url,
        json=payload,
        timeout=settings.external_model_api_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def infer_engineered_model(raw_text: str, cleaned_text: str, engineered_features: dict) -> dict:
    feature_order = list(engineered_features.keys())
    engineered_frame = pd.DataFrame([engineered_features])[feature_order]

    payload = {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "engineered_features": engineered_features,
        "engineered_features_frame": engineered_frame.to_dict(orient="records"),
        "feature_order": feature_order,
    }

    response = requests.post(
        settings.external_engineered_model_api_url,
        json=payload,
        timeout=settings.external_model_api_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()
