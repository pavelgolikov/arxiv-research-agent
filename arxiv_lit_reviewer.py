#!/usr/bin/env python3
from __future__ import annotations

import arxiv
import argparse
import os
import time
from pathlib import Path
from typing import TypeVar, TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


GEMINI_MODEL = "gemini-3.1-flash-lite"
T = TypeVar("T", bound=BaseModel)


class SearchPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=3)


class PaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_url: str
    entry_url: str


class ParsedPaper(BaseModel):
    arxiv_id: str
    text: str
    page_count: int = Field(ge=1)


class RelevanceDecision(BaseModel):
    arxiv_id: str
    is_relevant: bool
    score: int = Field(ge=1, le=5)
    reason: str


class PaperAnalysis(BaseModel):
    arxiv_id: str
    title: str
    research_problem: str
    method: str
    evidence: str
    findings: str
    limitations: str
    relevance_to_query: str


class LiteratureReview(BaseModel):
    title: str
    overview: str
    papers: list[PaperAnalysis]
    themes: list[str]
    gaps: list[str]
    suggested_reading_order: list[str]


class SearchNodeState(TypedDict, total=False):
    user_query: str
    max_results: int
    search_queries: list[str]
    found_papers: list[PaperMetadata]


def gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def generate_text(prompt: str) -> str:
    client = gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text or ""


def generate_structured(prompt: str, result_type: type[T]) -> T:
    client = gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            responseMimeType="application/json",
            responseSchema=result_type,
        ),
    )
    if response.parsed is not None:
        return result_type.model_validate(response.parsed)
    return result_type.model_validate_json(response.text or "")


def make_search_plan(user_query: str) -> SearchPlan:
    prompt = (
        "Convert this research question into 1 to 3 concise arXiv search queries. "
        "Use technical keywords that are likely to appear in paper titles or abstracts. "
        "Do not write explanations.\n\n"
        f"Research question: {user_query}"
    )
    return generate_structured(prompt, SearchPlan)


def search_arxiv(arxiv_query: str, max_results: int) -> list[PaperMetadata]:

    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=arxiv_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[PaperMetadata] = []
    seen_ids: set[str] = set()

    for result in client.results(search):
        arxiv_id = result.entry_id.rsplit("/", 1)[-1]
        if arxiv_id in seen_ids:
            continue

        seen_ids.add(arxiv_id)
        papers.append(
            PaperMetadata(
                arxiv_id=arxiv_id,
                title=" ".join(result.title.split()),
                authors=[
                    getattr(author, "name", str(author)) for author in result.authors
                ],
                abstract=" ".join(result.summary.split()),
                published=result.published.date().isoformat(),
                pdf_url=result.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                entry_url=result.entry_id,
            )
        )

    return papers


def search_node(state: SearchNodeState) -> SearchNodeState:
    user_query = state["user_query"]
    max_results = state.get("max_results", 10)
    search_plan = make_search_plan(user_query)
    arxiv_queries = [query.strip() for query in search_plan.queries if query.strip()]

    if not arxiv_queries:
        arxiv_queries = [user_query]

    found_papers: list[PaperMetadata] = []
    seen_ids: set[str] = set()

    for index, arxiv_query in enumerate(arxiv_queries):
        if len(found_papers) >= max_results:
            break
        if index > 0:
            time.sleep(3.0)

        remaining = max_results - len(found_papers)
        for paper in search_arxiv(arxiv_query, remaining):
            if paper.arxiv_id in seen_ids:
                continue
            seen_ids.add(paper.arxiv_id)
            found_papers.append(paper)

    return {
        "search_queries": arxiv_queries,
        "found_papers": found_papers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an arXiv literature reviewer and write a Markdown report."
    )
    parser.add_argument("--user-query", help="Research question or topic to review.")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--target-papers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("review.md"))
    args = parser.parse_args()

    load_dotenv()

    if args.max_results < 1:
        parser.error("--max-results must be at least 1.")
    if args.target_papers < 1:
        parser.error("--target-papers must be at least 1.")
    if args.target_papers > args.max_results:
        parser.error("--target-papers cannot be greater than --max-results.")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        parser.error("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
