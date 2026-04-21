from qdrant_client import QdrantClient

from config import settings


def _build_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=settings.qdrant_local_path)


def retrieve_embedding(query_vector, top_k: int):
    client = _build_client()
    hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload or {} for hit in hits]