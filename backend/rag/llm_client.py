from typing import List

from openai import OpenAI

from config import settings


SYSTEM_PROMPT = (
    "You are a support assistant. Answer clearly, be concise, and suggest the next action. "
    "If context is provided, ground your answer in it and mention relevant ticket IDs."
)


def generate_answer(ticket_text: str, context_snippets: List[str] | None = None) -> str:
    context_snippets = context_snippets or []
    context_block = "\n".join(context_snippets).strip()

    if not settings.openai_api_key:
        if context_block:
            return (
                "[Mocked LLM response with RAG] Based on similar tickets, check account status, "
                "retry the action, and escalate if issue persists."
            )
        return "[Mocked LLM response without RAG] Please provide more account and error details."

    user_prompt = f"""
Current user ticket:
{ticket_text}

Retrieved context (may be empty):
{context_block or 'No retrieved context'}

Write a support answer.
""".strip()

    client = OpenAI(api_key=settings.openai_api_key)
    completion = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content or ""
