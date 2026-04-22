from functools import lru_cache

from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient

from config import settings


router = APIRouter(prefix="/qdrant", tags=["qdrant"])


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=settings.qdrant_local_path)


@router.get("/health")
def health() -> dict:
    try:
        client = get_qdrant_client()
        client.get_collections()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"qdrant unavailable: {exc}") from exc

    return {"status": "ok", "client": "initialized", "db": "responded"}

