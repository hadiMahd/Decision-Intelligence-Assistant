import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

FEATURE_ORDER = [
    "sentiment_score",
    "has_cancel",
    "has_error",
    "has_problem",
    "exclamation_count",
    "has_down",
    "question_count",
    "has_refund",
    "has_issue",
    "has_broken",
]


def preprocess_raw_text(raw_text: str) -> tuple[str, dict]:
    text = str(raw_text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?.]", "", text)
    cleaned_text = re.sub(r"\s+", " ", text).strip()

    features = {
        "sentiment_score": analyzer.polarity_scores(cleaned_text)["compound"],
        "has_cancel": 1 if "cancel" in cleaned_text else 0,
        "has_error": 1 if "error" in cleaned_text else 0,
        "has_problem": 1 if "problem" in cleaned_text else 0,
        "exclamation_count": cleaned_text.count("!"),
        "has_down": 1 if "down" in cleaned_text else 0,
        "question_count": cleaned_text.count("?"),
        "has_refund": 1 if "refund" in cleaned_text else 0,
        "has_issue": 1 if "issue" in cleaned_text else 0,
        "has_broken": 1 if "broken" in cleaned_text else 0,
    }

    ordered_features = {name: features[name] for name in FEATURE_ORDER}
    return cleaned_text, ordered_features
