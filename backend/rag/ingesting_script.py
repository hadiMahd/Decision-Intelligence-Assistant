from __future__ import annotations

import logging
import zlib
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
logger = logging.getLogger(__name__)


def _ensure_collection(client) -> None:
    if not client.collection_exists(collection_name=settings.qdrant_collection):
        logger.info("Creating missing Qdrant collection: %s", settings.qdrant_collection)
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.qdrant_vector_size,
                distance=Distance.COSINE,
            ),
        )


def _stable_point_id(ticket_id: str) -> int:
    if ticket_id.isdigit():
        return int(ticket_id)
    return zlib.crc32(ticket_id.encode("utf-8"))


def ingest_text_to_qdrant(text: str, ticket_id: str | None = None, source: str = "manual_test") -> dict:
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("text must not be empty")

    normalized_id = ticket_id.strip() if ticket_id else f"manual-{zlib.crc32(normalized_text.encode('utf-8'))}"
    normalized_source = source.strip() if source else "manual_test"

    ensure_artifact_dirs()
    client = get_qdrant_client()
    _ensure_collection(client)

    vector = embed_texts([normalized_text])[0]
    point = PointStruct(
        id=_stable_point_id(normalized_id),
        vector=vector,
        payload={
            "id": normalized_id,
            "text": normalized_text,
            "source": normalized_source,
        },
    )

    client.upsert(collection_name=settings.qdrant_collection, points=[point])
    logger.info("Ingested single text into Qdrant: id=%s source=%s", normalized_id, normalized_source)

    return {
        "ingested": 1,
        "id": normalized_id,
        "source": normalized_source,
        "collection_name": settings.qdrant_collection,
    }


def ingest_csv_to_qdrant(max_rows: int = 20, reset_collection: bool = False) -> dict:
    csv_path = DATA_DIR / CSV_FILE
    if not csv_path.exists():
        logger.error("CSV file not found at path: %s", csv_path)
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    logger.info("Starting CSV ingest: file=%s max_rows=%s", csv_path, max_rows)
    ensure_artifact_dirs()
    df = pd.read_csv(csv_path).fillna("").head(max_rows)
    logger.info("CSV loaded: rows_read=%s", len(df))

    if ID_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Missing columns. Found: {list(df.columns)}. "
            f"Expected id={ID_COLUMN}, text={TEXT_COLUMN}."
        )

    client = get_qdrant_client()

    if reset_collection and client.collection_exists(collection_name=settings.qdrant_collection):
        logger.warning("Resetting Qdrant collection: %s", settings.qdrant_collection)
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
            logger.info("Skipping empty ingest batch at start_index=%s", start)
            continue

        valid_ids = [x[0] for x in valid_rows]
        valid_texts = [x[1] for x in valid_rows]

        vectors = embed_texts(valid_texts)

        points = []
        for ticket_id, text, vector in zip(valid_ids, valid_texts, vectors):
            point_id = _stable_point_id(ticket_id)
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
        logger.info(
            "Upserted batch: start_index=%s batch_size=%s total_ingested=%s",
            start,
            len(points),
            rows_ingested,
        )

    logger.info("CSV ingest finished: rows_read=%s rows_ingested=%s", len(df), rows_ingested)
    return {
        "file_path": str(csv_path),
        "rows_read": len(df),
        "rows_ingested": rows_ingested,
        "collection_name": settings.qdrant_collection,
    }