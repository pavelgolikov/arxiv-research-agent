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


class DraftEvidence(BaseModel):
    """One citation as proposed by the model, before validation."""

    chunk_id: str
    excerpt: str


class DraftClaim(BaseModel):
    """One claim as proposed by the model, before validation."""

    text: str
    evidence: list[DraftEvidence]


class FacetDraft(BaseModel):
    """The model's unvalidated answer for a single analysis facet."""

    claims: list[DraftClaim] = Field(max_length=4)


class EvidenceRef(BaseModel):
    """A citation whose chunk, paper, and excerpt have all been verified."""

    chunk_id: str
    arxiv_id: str
    page_number: int = Field(ge=1)
    excerpt: str


class SupportedClaim(BaseModel):
    """A claim that retains at least one verified citation."""

    text: str
    evidence: list[EvidenceRef] = Field(min_length=1)


class GroundedAnalysis(BaseModel):
    """Validated per-facet claims for one paper."""

    arxiv_id: str
    title: str
    claims: dict[str, list[SupportedClaim]]
    dropped_claims: int = 0
    dropped_evidence: int = 0

    @property
    def is_partial(self) -> bool:
        """Report whether any proposed claim or citation failed validation."""

        return bool(self.dropped_claims or self.dropped_evidence)

    @property
    def supported_claim_count(self) -> int:
        """Count the claims that survived validation."""

        return sum(len(claims) for claims in self.claims.values())


class LiteratureReview(BaseModel):
    title: str
    overview: str
    papers: list[GroundedAnalysis]
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
    chosen_papers: dict[str, GroundedAnalysis]
    markdown: str
    status: str
