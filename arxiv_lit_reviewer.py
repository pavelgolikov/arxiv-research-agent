#!/usr/bin/env python3
from __future__ import annotations

import arxiv
import argparse
import os
import time
from pathlib import Path
from typing import TypeVar, TypedDict

import fitz
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


GEMINI_MODEL = "gemini-3.1-flash-lite"
RELEVANCE_TEXT_CHARS = 12000
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
    experimental_setup: str
    main_findings: str
    limitations: str
    relevance_to_query: str


class LiteratureReview(BaseModel):
    title: str
    overview: str
    papers: list[PaperAnalysis]
    themes: list[str]
    gaps: list[str]
    suggested_reading_order: list[str]


class ReviewerState(TypedDict, total=False):
    user_query: str
    max_results: int
    target_papers: int
    search_queries: list[str]
    found_papers: list[PaperMetadata]
    current_paper_index: int
    parsed_papers: dict[str, ParsedPaper]
    relevance_decisions: dict[str, RelevanceDecision]
    chosen_papers: dict[str, PaperAnalysis]


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


def search_node(state: ReviewerState) -> ReviewerState:
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


def download_parse_node(state: ReviewerState) -> ReviewerState:
    found_papers = state["found_papers"]
    current_paper_index = state.get("current_paper_index", 0)
    paper = found_papers[current_paper_index]
    parsed_papers = dict(state.get("parsed_papers", {}))

    if paper.arxiv_id in parsed_papers:
        return {"parsed_papers": parsed_papers}

    response = httpx.get(paper.pdf_url, follow_redirects=True, timeout=60)
    response.raise_for_status()

    with fitz.open(stream=response.content, filetype="pdf") as document:
        text = "\n".join(page.get_text() for page in document)
        page_count = document.page_count

    parsed_papers[paper.arxiv_id] = ParsedPaper(
        arxiv_id=paper.arxiv_id,
        text=text,
        page_count=page_count,
    )
    return {"parsed_papers": parsed_papers}


def relevance_eval_node(state: ReviewerState) -> ReviewerState:
    current_paper_index = state.get("current_paper_index", 0)
    paper = state["found_papers"][current_paper_index]
    parsed_paper = state["parsed_papers"][paper.arxiv_id]
    relevance_decisions = dict(state.get("relevance_decisions", {}))

    prompt = (
        "Decide whether this arXiv paper is relevant to the user's research query.\n"
        "Use this score rubric:\n"
        "1 = unrelated.\n"
        "2 = weakly related.\n"
        "3 = possibly useful but not central.\n"
        "4 = clearly relevant.\n"
        "5 = highly relevant and should be included unless the paper is low quality.\n"
        "Set is_relevant to true only when score is 4 or 5.\n\n"
        f"User query: {state['user_query']}\n\n"
        f"arXiv ID: {paper.arxiv_id}\n"
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors)}\n"
        f"Published: {paper.published}\n"
        f"Abstract: {paper.abstract}\n\n"
        f"Paper text preview:\n{parsed_paper.text[:RELEVANCE_TEXT_CHARS]}"
    )
    decision = generate_structured(prompt, RelevanceDecision)
    decision = decision.model_copy(
        update={"arxiv_id": paper.arxiv_id, "is_relevant": decision.score >= 4}
    )
    relevance_decisions[paper.arxiv_id] = decision

    return {"relevance_decisions": relevance_decisions}


def route_after_relevance_eval(state: ReviewerState) -> str:
    current_paper_index = state.get("current_paper_index", 0)
    paper = state["found_papers"][current_paper_index]
    decision = state["relevance_decisions"][paper.arxiv_id]

    if decision.is_relevant:
        return "extract_core"
    return "advance_paper"


def advance_paper_node(state: ReviewerState) -> ReviewerState:
    return {"current_paper_index": state.get("current_paper_index", 0) + 1}


def extract_core_node(state: ReviewerState) -> ReviewerState:
    current_paper_index = state.get("current_paper_index", 0)
    paper = state["found_papers"][current_paper_index]
    parsed_paper = state["parsed_papers"][paper.arxiv_id]
    chosen_papers = dict(state.get("chosen_papers", {}))

    if paper.arxiv_id in chosen_papers:
        return {"chosen_papers": chosen_papers}

    prompt = (
        "Extract structured literature-review notes from this arXiv paper.\n"
        "Use the full paper text when possible. Be specific and concise.\n\n"
        f"User query: {state['user_query']}\n\n"
        f"arXiv ID: {paper.arxiv_id}\n"
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors)}\n"
        f"Published: {paper.published}\n"
        f"Abstract: {paper.abstract}\n\n"
        f"Full paper text:\n{parsed_paper.text}"
    )
    analysis = generate_structured(prompt, PaperAnalysis)
    analysis = analysis.model_copy(update={"arxiv_id": paper.arxiv_id, "title": paper.title})
    chosen_papers[paper.arxiv_id] = analysis

    return {"chosen_papers": chosen_papers}


def route_after_extract_core(state: ReviewerState) -> str:
    chosen_papers = state.get("chosen_papers", {})
    target_papers = state.get("target_papers", 4)

    if len(chosen_papers) >= target_papers:
        return "write_markdown"
    return "advance_paper"


def route_after_advance_paper(state: ReviewerState) -> str:
    current_paper_index = state.get("current_paper_index", 0)

    if current_paper_index < len(state.get("found_papers", [])):
        return "download_parse"
    return "write_markdown"


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
