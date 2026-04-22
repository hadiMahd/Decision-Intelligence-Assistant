from __future__ import annotations

from pathlib import Path

import pandas as pd
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import ensure_artifact_dirs, settings
from rag.embed_query import embed_texts
from routers.qdrant_db import get_qdrant_client


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_FILE = "howto_tickets.csv"
ID_COLUMN = "id"
TEXT_COLUMN = "text"
DEFAULT_BATCH_SIZE = 20


def _ensure_collection(client) -> None:
    if not client.collection_exists(collection_name=settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.qdrant_vector_size,
                distance=Distance.COSINE,
            ),
        )


def ingest_csv_to_qdrant(max_rows: int = 20, reset_collection: bool = False) -> dict:
    csv_path = DATA_DIR / CSV_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    ensure_artifact_dirs()
    df = pd.read_csv(csv_path).fillna("").head(max_rows)

    if ID_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Missing columns. Found: {list(df.columns)}. "
            f"Expected id={ID_COLUMN}, text={TEXT_COLUMN}."
        )

    client = get_qdrant_client()

    if reset_collection and client.collection_exists(collection_name=settings.qdrant_collection):
        client.delete_collection(collection_name=settings.qdrant_collection)

    _ensure_collection(client)

    ids = [str(v).strip() for v in df[ID_COLUMN].tolist()]
    texts = [str(v).strip() for v in df[TEXT_COLUMN].tolist()]

    rows_ingested = 0
    for start in range(0, len(texts), DEFAULT_BATCH_SIZE):
        batch_ids = ids[start : start + DEFAULT_BATCH_SIZE]
        batch_texts = texts[start : start + DEFAULT_BATCH_SIZE]

        valid_rows = [(i, t) for i, t in zip(batch_ids, batch_texts) if t]
        if not valid_rows:
            continue

        valid_ids = [x[0] for x in valid_rows]
        valid_texts = [x[1] for x in valid_rows]

        vectors = embed_texts(valid_texts)

        points = []
        for ticket_id, text, vector in zip(valid_ids, valid_texts, vectors):
            point_id = int(ticket_id) if ticket_id.isdigit() else abs(hash(ticket_id))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "id": ticket_id,
                        "text": text,
                        "source": CSV_FILE,
                    },
                )
            )

        client.upsert(collection_name=settings.qdrant_collection, points=points)
        rows_ingested += len(points)

    return {
        "file_path": str(csv_path),
        "rows_read": len(df),
        "rows_ingested": rows_ingested,
        "collection_name": settings.qdrant_collection,
    }