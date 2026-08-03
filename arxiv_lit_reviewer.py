#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from pydantic import BaseModel, Field


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
    score: float = Field(ge=0.0, le=1.0)
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an arXiv literature reviewer and write a Markdown report."
    )
    parser.add_argument("query", help="Research question or topic to review.")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--target-papers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("review.md"))
    args = parser.parse_args()

    if args.max_results < 1:
        parser.error("--max-results must be at least 1.")
    if args.target_papers < 1:
        parser.error("--target-papers must be at least 1.")
    if args.target_papers > args.max_results:
        parser.error("--target-papers cannot be greater than --max-results.")
    if not os.getenv("GEMINI_API_KEY"):
        parser.error("GEMINI_API_KEY is not set.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
