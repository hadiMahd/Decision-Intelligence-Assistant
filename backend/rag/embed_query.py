import math
from typing import List

from openai import OpenAI

from config import settings


def _local_fallback_embedding(text: str, dim: int = 64) -> List[float]:
    # Cheap deterministic fallback so local tests still work without API keys.
    vec = [0.0] * dim
    for idx, ch in enumerate(text.lower()):
        bucket = (ord(ch) + idx) % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not settings.openai_api_key:
        return [_local_fallback_embedding(t) for t in texts]

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embedding_dimension() -> int:
    if settings.openai_api_key:
        return settings.qdrant_vector_size
    return 64
