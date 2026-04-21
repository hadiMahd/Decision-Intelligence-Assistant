from functools import lru_cache
from pathlib import Path
from typing import List

from openai import OpenAI

from config import settings

PROMPT_FILE = Path(__file__).resolve().parents[1] / "prompts" / "context_only_support_prompt.txt"
NO_CONTEXT_REPLY = "couldnt find relevent results and i cant help u"


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    if PROMPT_FILE.exists():
        content = PROMPT_FILE.read_text(encoding="utf-8").strip()
        if content:
            return content
    return (
        "You are a customer support assistant. "
        "Use ONLY the retrieved context to answer. "
        f"If context is missing, reply exactly with: {NO_CONTEXT_REPLY}"
    )


def generate_answer(ticket_text: str, context_snippets: List[str] | None = None) -> str:
    context_snippets = [snippet.strip() for snippet in (context_snippets or []) if snippet and snippet.strip()]
    if not context_snippets:
        return NO_CONTEXT_REPLY

    context_block = "\n".join(context_snippets)
    user_prompt = f"""
Current user ticket:
{ticket_text}

Retrieved context:
{context_block}

Answer the ticket using ONLY the retrieved context.
If the context is not enough, reply exactly with:
{NO_CONTEXT_REPLY}
""".strip()

    if not settings.openai_api_key:
        return (
            "[Mocked LLM response with RAG] "
            "Context was provided, so the answer should be grounded only in that context."
        )

    completion = get_openai_client().chat.completions.create(
        model=settings.answering_model,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    return completion.choices[0].message.content or ""
