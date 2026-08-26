"""Gemini access through LangChain, including structured generation helpers."""

import os
import threading
from typing import TypeVar

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

GEMINI_MODEL = "gemini-3.1-flash-lite"
T = TypeVar("T", bound=BaseModel)

# Token counts for every call made since the last reset, so the cost of one run can be
# measured. Branches fan out across threads, so appends are locked. This is read by a
# throwaway measurement script, not by the pipeline: nothing here changes what a run
# does, and leaving it in place means the figure can be measured again after a change
# rather than rebuilt from scratch.
_usage: list[tuple[int, int]] = []
_usage_lock = threading.Lock()


def record_tokens(usage_metadata: dict | None) -> None:
    """Store the input and output token counts reported for one call."""

    if not usage_metadata:
        return
    with _usage_lock:
        _usage.append(
            (
                int(usage_metadata.get("input_tokens", 0)),
                int(usage_metadata.get("output_tokens", 0)),
            )
        )


def reset_usage() -> None:
    """Discard everything recorded so far, so one run can be measured alone."""

    with _usage_lock:
        _usage.clear()


def usage_totals() -> dict[str, int]:
    """Return the calls and token counts recorded since the last reset."""

    with _usage_lock:
        recorded = list(_usage)

    return {
        "calls": len(recorded),
        "input_tokens": sum(item[0] for item in recorded),
        "output_tokens": sum(item[1] for item in recorded),
    }


def gemini_llm() -> ChatGoogleGenerativeAI:
    """Create a LangChain Gemini chat model from the configured API key."""

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=api_key)


def generate_text(prompt: str) -> str:
    """Send a plain-text prompt to Gemini and return text."""

    response = gemini_llm().invoke(prompt)
    record_tokens(getattr(response, "usage_metadata", None))
    if isinstance(response.content, str):
        return response.content

    text_parts = []
    for content_item in response.content:
        if isinstance(content_item, str):
            text_parts.append(content_item)
        elif isinstance(content_item, dict) and content_item.get("type") == "text":
            text_parts.append(str(content_item.get("text", "")))

    return "\n".join(text_parts)


def generate_structured(prompt: str, result_type: type[T]) -> T:
    """Send a prompt to Gemini and validate the structured response.

    `include_raw` is what makes the token counts reachable: without it LangChain returns
    the parsed object alone and discards the message carrying `usage_metadata`. It also
    wraps the parser in a fallback, so a response that violates the schema is returned
    under `parsing_error` instead of being raised. Callers rely on that exception to
    record a typed failure describing what the model got wrong, so it is re-raised here.
    """

    result = gemini_llm().with_structured_output(result_type, include_raw=True).invoke(prompt)

    if result["parsing_error"] is not None:
        raise result["parsing_error"]

    record_tokens(getattr(result["raw"], "usage_metadata", None))
    return result_type.model_validate(result["parsed"])
