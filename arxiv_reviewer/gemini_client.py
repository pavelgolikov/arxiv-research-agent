"""Gemini access through LangChain, including structured generation helpers."""

import os
from typing import TypeVar

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

GEMINI_MODEL = "gemini-3.1-flash-lite"
T = TypeVar("T", bound=BaseModel)


def gemini_llm() -> ChatGoogleGenerativeAI:
    """Create a LangChain Gemini chat model from the configured API key."""

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=api_key)


def generate_text(prompt: str) -> str:
    """Send a plain-text prompt to Gemini and return text."""

    response = gemini_llm().invoke(prompt)
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
    """Send a prompt to Gemini and validate the structured response."""

    response = gemini_llm().with_structured_output(result_type).invoke(prompt)
    return result_type.model_validate(response)
