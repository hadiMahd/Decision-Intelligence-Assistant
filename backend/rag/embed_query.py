from functools import lru_cache
from typing import List

from openai import OpenAI
from openai import OpenAIError

from config import settings


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAI:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for embedding generation.")
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        raise ValueError("At least one input text is required for embeddings.")

    try:
        response = get_embedding_client().embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
    except OpenAIError as exc:
        raise RuntimeError(f"Embedding request failed: {exc}") from exc

    return [item.embedding for item in response.data]


def embedding_dimension() -> int:
    return settings.qdrant_vector_size
