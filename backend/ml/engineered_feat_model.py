import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

from ml.preprocess import FEATURE_ORDER

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "final_urgency_model.pkl"


@lru_cache(maxsize=1)
def _load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Engineered model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def infer_engineered(engineered_features: dict[str, float | int]) -> tuple[Any, float | None]:
    if not engineered_features:
        raise ValueError("Engineered features dictionary cannot be empty.")

    model = _load_model()
    ordered_row = [[engineered_features[name] for name in FEATURE_ORDER]]

    prediction = model.predict(ordered_row)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(ordered_row)[0]
        confidence = float(max(probas))

    return prediction, confidence
