import logging

from config import settings
from routers.qdrant_db import get_qdrant_client


logger = logging.getLogger(__name__)


def retrieve_embedding(query_vector, top_k: int):
    logger.info(
        "Querying Qdrant: collection=%s top_k=%s vector_dim=%s",
        settings.qdrant_collection,
        top_k,
        len(query_vector) if hasattr(query_vector, "__len__") else "unknown",
    )
    client = get_qdrant_client()
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
    )

    points = getattr(results, "points", results)
    payloads = [point.payload or {} for point in points]
    ids = [str(payload.get("id") or "unknown") for payload in payloads]
    logger.info("Qdrant query returned %s points with ids=%s", len(payloads), ids)
    return payloads