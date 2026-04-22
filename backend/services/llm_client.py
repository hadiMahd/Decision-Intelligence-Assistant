from functools import lru_cache
from typing import List

from openai import OpenAI

from config import settings
from prompts.plain_answer import PLAIN_SYSTEM_PROMPT

NO_CONTEXT_REPLY = "couldnt find relevent results and i cant help u"


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    return PLAIN_SYSTEM_PROMPT


def _call_llm(system_prompt: str, user_prompt: str, has_context: bool) -> str:
    if not settings.openai_api_key:
        if has_context:
            return (
                "[Mocked LLM response with RAG] "
                "Context was provided, so the answer should be grounded only in that context."
            )
        return "[Mocked LLM response without RAG] Please answer based on the ticket only."

    completion = get_openai_client().chat.completions.create(
        model=settings.answering_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content or ""


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
If the context is not enough, reply exactly with:
{NO_CONTEXT_REPLY}
""".strip()
    return _call_llm(system_prompt, user_prompt, has_context=True)


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
