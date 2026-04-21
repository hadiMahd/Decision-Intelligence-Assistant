from fastapi import APIRouter

from config import settings
from rag.embed_query import embed_texts
from rag.llm_client import generate_answer
from rag.search_db import retrieve_embedding
from schemas.rag_schemas import RAGCompareRequest, RAGCompareResponse, RAGScore

router = APIRouter(prefix="/rag", tags=["rag"])


def _word_overlap(a: str, b: str) -> float:
    a_set = {w for w in a.lower().split() if len(w) > 2}
    b_set = {w for w in b.lower().split() if len(w) > 2}
    if not a_set:
        return 0.0
    return round(len(a_set.intersection(b_set)) / len(a_set), 3)


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
