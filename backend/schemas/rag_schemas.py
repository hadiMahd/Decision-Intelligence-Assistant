from pydantic import BaseModel, Field


class RAGCompareRequest(BaseModel):
    ticket_text: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class RAGScore(BaseModel):
    context_overlap_score: float
    grounding_gain_score: float


class RAGCompareResponse(BaseModel):
    no_rag_answer: str
    rag_answer: str
    retrieved_tickets: list[dict]
    scores: RAGScore