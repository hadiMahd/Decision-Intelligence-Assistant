from __future__ import annotations

import logging
from typing import Any

from prompts.grounded_answer import GROUNDED_SYSTEM_PROMPT
from services.llm_client import generate_grounded_answer, generate_plain_answer


logger = logging.getLogger(__name__)


def _chunk_to_context_snippet(chunk: dict[str, Any]) -> str:
    chunk_id = chunk.get("id") or "unknown"
    text = str(chunk.get("text") or "").strip()
    source = str(chunk.get("source") or "").strip()

    if source and text:
        return f"[{chunk_id}] {text} (source: {source})"
    if text:
        return f"[{chunk_id}] {text}"
    if source:
        return f"[{chunk_id}] source: {source}"
    return f"[{chunk_id}]"


def _grounded_prompt(context_snippets: list[str]) -> str:
    context_block = "\n".join(context_snippets)
    return GROUNDED_SYSTEM_PROMPT.format(context=context_block)


def get_grounded_and_plain_answers(
    ticket_text: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, str]:
    logger.info("Generating plain and grounded answers: retrieved_chunks=%s", len(retrieved_chunks))
    plain_answer = generate_plain_answer(ticket_text=ticket_text)

    context_snippets = [
        _chunk_to_context_snippet(chunk)
        for chunk in retrieved_chunks
        if isinstance(chunk, dict)
    ]
    grounded_prompt = _grounded_prompt(context_snippets)
    grounded_answer = generate_grounded_answer(
        ticket_text=ticket_text,
        system_prompt=grounded_prompt,
        context_snippets=context_snippets,
    )
    logger.info("Grounded answer generated with %s context snippets", len(context_snippets))

    return {
        "plain_answer": plain_answer,
        "grounded_answer": grounded_answer,
    }
