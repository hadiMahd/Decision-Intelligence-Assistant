import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

from ml.preprocess import preprocess_raw_text

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_model_vectorizer.pkl"
MODEL_PATH = ARTIFACTS_DIR / "tfidf_urgency_model.pkl"


@lru_cache(maxsize=1)
def _load_vectorizer() -> Any:
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {VECTORIZER_PATH}")
    with open(VECTORIZER_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"TF-IDF model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def infer_tfidf(raw_text: str, cleaned_text: str | None = None) -> tuple[Any, float | None]:
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text cannot be empty.")

    normalized_text = cleaned_text
    if normalized_text is None:
        normalized_text, _ = preprocess_raw_text(raw_text)

    vectorizer = _load_vectorizer()
    model = _load_model()

    tfidf_vector = vectorizer.transform([normalized_text])
    prediction = model.predict(tfidf_vector)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(tfidf_vector)[0]
        confidence = float(max(probabilities))

    return prediction, confidence
