from fastapi import APIRouter, HTTPException

from answering_llm.llm_client import generate_answer
from config import settings
from rag.embed_query import embed_texts
from rag.ingesting_script import ingest_csv_to_qdrant
from rag.search_db import retrieve_embedding
from schemas.rag_schemas import RAGCompareRequest, RAGCompareResponse, RAGScore

router = APIRouter(prefix="/rag", tags=["rag"])


def _word_overlap(a: str, b: str) -> float:
    a_set = {w for w in a.lower().split() if len(w) > 2}
    b_set = {w for w in b.lower().split() if len(w) > 2}
    if not a_set:
        return 0.0
    return round(len(a_set.intersection(b_set)) / len(a_set), 3)


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

    no_rag_answer = generate_answer(ticket_text=payload.ticket_text, context_snippets=[])

    snippets = [
        f"[{t.get('ticket_id', 'unknown')}] {t.get('title', '')}. Resolution: {t.get('resolution', '')}"
        for t in retrieved
    ]
    rag_answer = generate_answer(ticket_text=payload.ticket_text, context_snippets=snippets)

    context_text = " ".join(snippets)
    rag_overlap = _word_overlap(rag_answer, context_text)
    no_rag_overlap = _word_overlap(no_rag_answer, context_text)

    scores = RAGScore(
        context_overlap_score=rag_overlap,
        grounding_gain_score=round(rag_overlap - no_rag_overlap, 3),
    )

    return RAGCompareResponse(
        no_rag_answer=no_rag_answer,
        rag_answer=rag_answer,
        retrieved_tickets=retrieved,
        scores=scores,
    )
