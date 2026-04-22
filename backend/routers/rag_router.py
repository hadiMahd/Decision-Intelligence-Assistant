from fastapi import APIRouter, HTTPException

from config import settings
from rag.embed_query import embed_texts
from rag.ingesting_script import ingest_csv_to_qdrant
from rag.search_db import retrieve_embedding
from services.llm_grounding import get_grounded_and_plain_answers
from schemas.rag_schemas import RAGCompareRequest, RAGCompareResponse, RAGSearchRequest, RAGSearchResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ingest-csv")
def ingest_csv(payload: dict | None = None) -> dict:
    max_rows = 20
    if payload and "max_rows" in payload:
        try:
            max_rows = int(payload.get("max_rows", 20))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid max_rows value: {exc}") from exc

    if max_rows < 1:
        raise HTTPException(status_code=400, detail="max_rows must be >= 1")

    try:
        result = ingest_csv_to_qdrant(
            max_rows=max_rows,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return result


@router.post("/compare", response_model=RAGCompareResponse)
def compare_with_and_without_rag(payload: RAGCompareRequest) -> RAGCompareResponse:
    query_vector = embed_texts([payload.ticket_text])[0]
    retrieved = retrieve_embedding(query_vector=query_vector, top_k=payload.top_k or settings.top_k)

    answers = get_grounded_and_plain_answers(
        ticket_text=payload.ticket_text,
        retrieved_chunks=retrieved,
    )
    no_rag_answer = answers["plain_answer"]
    rag_answer = answers["grounded_answer"]

    return RAGCompareResponse(
        no_rag_answer=no_rag_answer,
        rag_answer=rag_answer,
        retrieved_tickets=retrieved,
    )


@router.post("/search", response_model=RAGSearchResponse)
def search_rag(payload: RAGSearchRequest) -> RAGSearchResponse:
    query_vector = embed_texts([payload.query])[0]
    retrieved = retrieve_embedding(query_vector=query_vector, top_k=payload.top_k or settings.top_k)

    return RAGSearchResponse(
        query=payload.query,
        top_k=payload.top_k,
        results=retrieved,
    )

