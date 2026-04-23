import logging
from functools import lru_cache
from typing import List
from typing import Literal

from openai import OpenAI

from config import settings
from prompts.plain_answer import PLAIN_SYSTEM_PROMPT

NO_CONTEXT_REPLY = "couldnt find relevent results and i cant help u"
logger = logging.getLogger(__name__)
MAX_LOG_TEXT_CHARS = 280


def _clip_for_log(text: str, limit: int = MAX_LOG_TEXT_CHARS) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def _normalize_no_context_answer(answer: str) -> str:
    normalized_target = " ".join(NO_CONTEXT_REPLY.lower().split())
    normalized_answer = " ".join((answer or "").lower().split()).strip(" .!?'\"")
    if normalized_target == normalized_answer:
        return NO_CONTEXT_REPLY
    return (answer or "").strip()


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    return PLAIN_SYSTEM_PROMPT


def _call_llm(system_prompt: str, user_prompt: str, has_context: bool) -> str:
    logger.info(
        "LLM call started: model=%s has_context=%s user_prompt=%s",
        settings.answering_model,
        has_context,
        _clip_for_log(user_prompt),
    )
    if not settings.openai_api_key:
        if has_context:
            logger.warning("LLM call mocked because OPENAI_API_KEY is not set")
            return (
                "[Mocked LLM response with RAG] "
                "Context was provided, so the answer should be grounded only in that context."
            )
        logger.warning("LLM call mocked because OPENAI_API_KEY is not set")
        return "[Mocked LLM response without RAG] Please answer based on the ticket only."

    completion = get_openai_client().chat.completions.create(
        model=settings.answering_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    answer = completion.choices[0].message.content or ""
    logger.info("LLM call completed: has_context=%s answer=%s", has_context, _clip_for_log(answer))
    return answer


def generate_plain_answer(ticket_text: str) -> str:
    user_prompt = f"""
Current user ticket:
{ticket_text}

Answer the ticket clearly and concisely.
If details are missing, ask one short follow-up question.
""".strip()
    return _call_llm(get_system_prompt(), user_prompt, has_context=False)


def generate_grounded_answer(ticket_text: str, context_snippets: List[str], system_prompt: str) -> str:
    cleaned_snippets = [snippet.strip() for snippet in context_snippets if snippet and snippet.strip()]
    if not cleaned_snippets:
        return NO_CONTEXT_REPLY

    context_block = "\n".join(cleaned_snippets)
    user_prompt = f"""
Current user ticket:
{ticket_text}

Retrieved context:
{context_block}

Answer the ticket using ONLY the retrieved context.
If the context includes part of the answer, provide that part clearly and state what is missing.
Only if there is no relevant information in the retrieved context, reply exactly with:
{NO_CONTEXT_REPLY}
""".strip()
    answer = _call_llm(system_prompt, user_prompt, has_context=True)
    return _normalize_no_context_answer(answer)


def generate_answer(
    ticket_text: str,
    context_snippets: List[str] | None = None,
    system_prompt: str | None = None,
) -> str:
    if context_snippets:
        return generate_grounded_answer(
            ticket_text=ticket_text,
            context_snippets=context_snippets,
            system_prompt=system_prompt or get_system_prompt(),
        )
    return generate_plain_answer(ticket_text=ticket_text)


def classify_ticket_urgency(ticket_text: str) -> Literal["urgent", "not_urgent"]:
    lowered = ticket_text.lower()
    heuristic_urgent_terms = (
        "down",
        "outage",
        "broken",
        "cannot",
        "can't",
        "cant",
        "urgent",
        "asap",
        "fails",
        "failing",
        "error",
    )

    if not settings.openai_api_key:
        prediction = "urgent" if any(term in lowered for term in heuristic_urgent_terms) else "not_urgent"
        logger.warning("Urgency classification is mocked because OPENAI_API_KEY is not set: prediction=%s", prediction)
        return prediction

    system_prompt = (
        "You are a strict classifier for support ticket urgency. "
        "Respond with exactly one label: urgent or not_urgent. "
        "No explanation and no extra words."
    )
    user_prompt = f"Ticket:\n{ticket_text}\n\nLabel:"
    raw = _call_llm(system_prompt=system_prompt, user_prompt=user_prompt, has_context=False).strip().lower()

    if "not_urgent" in raw:
        prediction = "not_urgent"
    elif raw == "urgent" or raw.startswith("urgent"):
        prediction = "urgent"
    else:
        prediction = "urgent" if any(term in lowered for term in heuristic_urgent_terms) else "not_urgent"
        logger.warning("Unexpected urgency label from LLM (%s); fallback prediction=%s", _clip_for_log(raw), prediction)

    logger.info("LLM urgency classification: prediction=%s", prediction)
    return prediction
