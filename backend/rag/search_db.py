from config import settings
from routers.qdrant_db import get_qdrant_client

def retrieve_embedding(query_vector, top_k: int):
    client = get_qdrant_client()
    hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload or {} for hit in hits]