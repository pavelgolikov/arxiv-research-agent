"""Typed state and structured model outputs for the literature reviewer."""

from typing import TypedDict

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


class ParsedPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str


class ParsedPaper(BaseModel):
    arxiv_id: str
    pages: list[ParsedPage]
    page_count: int = Field(ge=1)

    @property
    def full_text(self) -> str:
        """Join every parsed page into a single string."""

        return "\n".join(page.text for page in self.pages)


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
    output: str
    thread_id: str
    data_dir: str
    retriever_kind: str
    top_k: int
    fetch_k: int
    multi_query: bool
    search_queries: list[str]
    found_papers: list[PaperMetadata]
    current_paper_index: int
    parsed_papers: dict[str, ParsedPaper]
    chunk_counts: dict[str, int]
    relevance_decisions: dict[str, RelevanceDecision]
    chosen_papers: dict[str, PaperAnalysis]
    markdown: str
    status: str
