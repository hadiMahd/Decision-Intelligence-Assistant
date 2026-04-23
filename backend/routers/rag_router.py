import logging
from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException

from config import settings
from rag.embed_query import embed_texts
from rag.ingesting_script import ingest_csv_to_qdrant, ingest_text_to_qdrant
from rag.search_db import retrieve_embedding
from services.llm_grounding import get_grounded_and_plain_answers
from schemas.rag_schemas import (
    RAGCompareRequest,
    RAGCompareResponse,
    RAGIngestRequest,
    RAGIngestTextRequest,
    RAGSearchRequest,
    RAGSearchResponse,
)

router = APIRouter(prefix="/rag", tags=["rag"])
logger = logging.getLogger(__name__)
MAX_LOG_TEXT_CHARS = 280


def _clip_for_log(text: str, limit: int = MAX_LOG_TEXT_CHARS) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def _retrieved_ids(chunks: list[dict]) -> list[str]:
    return [str(chunk.get("id") or "unknown") for chunk in chunks]


@router.post("/ingest-csv")
def ingest_csv(payload: RAGIngestRequest = Body(default_factory=RAGIngestRequest)) -> dict:
    logger.info("/rag/ingest-csv called")

    try:
        result = ingest_csv_to_qdrant(
            max_rows=payload.max_rows,
        )
    except FileNotFoundError as exc:
        logger.exception("Ingest failed because CSV file was not found")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        logger.exception("Ingest failed due to invalid CSV schema")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during ingest")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    logger.info(
        "Ingest completed: rows_read=%s rows_ingested=%s collection=%s",
        result.get("rows_read"),
        result.get("rows_ingested"),
        result.get("collection_name"),
    )
    return result


@router.post("/ingest-text")
def ingest_text(payload: RAGIngestTextRequest) -> dict:
    logger.info("/rag/ingest-text called: id=%s source=%s text=%s", payload.id, payload.source, _clip_for_log(payload.text))

    try:
        result = ingest_text_to_qdrant(
            text=payload.text,
            ticket_id=payload.id,
            source=payload.source,
        )
    except ValueError as exc:
        logger.exception("Ingest text failed due to invalid payload")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during ingest-text")
        raise HTTPException(status_code=500, detail=f"Ingest text failed: {exc}") from exc

    logger.info(
        "Ingest-text completed: id=%s source=%s collection=%s",
        result.get("id"),
        result.get("source"),
        result.get("collection_name"),
    )
    return result


@router.post("/compare", response_model=RAGCompareResponse)
def compare_with_and_without_rag(payload: RAGCompareRequest) -> RAGCompareResponse:
    request_id = uuid4().hex[:8]
    logger.info(
        "/rag/compare called: request_id=%s top_k=%s ticket_text=%s",
        request_id,
        payload.top_k or settings.top_k,
        _clip_for_log(payload.ticket_text),
    )
    query_vector = embed_texts([payload.ticket_text])[0]
    retrieved = retrieve_embedding(query_vector=query_vector, top_k=payload.top_k or settings.top_k)
    logger.info(
        "/rag/compare retrieved: request_id=%s count=%s ids=%s",
        request_id,
        len(retrieved),
        _retrieved_ids(retrieved),
    )

    answers = get_grounded_and_plain_answers(
        ticket_text=payload.ticket_text,
        retrieved_chunks=retrieved,
    )
    no_rag_answer = answers["plain_answer"]
    rag_answer = answers["grounded_answer"]
    logger.info(
        "/rag/compare outputs: request_id=%s plain=%s grounded=%s",
        request_id,
        _clip_for_log(no_rag_answer),
        _clip_for_log(rag_answer),
    )

    return RAGCompareResponse(
        no_rag_answer=no_rag_answer,
        rag_answer=rag_answer,
        retrieved_tickets=retrieved,
    )


@router.post("/search", response_model=RAGSearchResponse)
def search_rag(payload: RAGSearchRequest) -> RAGSearchResponse:
    request_id = uuid4().hex[:8]
    logger.info(
        "/rag/search called: request_id=%s top_k=%s query=%s",
        request_id,
        payload.top_k or settings.top_k,
        _clip_for_log(payload.query),
    )
    query_vector = embed_texts([payload.query])[0]
    retrieved = retrieve_embedding(query_vector=query_vector, top_k=payload.top_k or settings.top_k)
    logger.info(
        "/rag/search returned: request_id=%s count=%s ids=%s",
        request_id,
        len(retrieved),
        _retrieved_ids(retrieved),
    )

    return RAGSearchResponse(
        query=payload.query,
        top_k=payload.top_k,
        results=retrieved,
    )

