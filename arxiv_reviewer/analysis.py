"""Candidate screening and grounded per-facet analysis branches."""

from pathlib import Path

from langchain_core.documents import Document

from .failures import describe, with_retries
from .gemini_client import generate_structured
from .rag import DEFAULT_FETCH_K, DEFAULT_TOP_K, chunk_pages, get_retriever, index_paper
from .retrieval import fetch_parsed_paper
from .review_types import (
    AnalysisOutcome,
    AnalyzeTask,
    DraftClaim,
    EvidenceRef,
    FacetDraft,
    GroundedAnalysis,
    RelevanceDecision,
    ReviewerState,
    ScreenOutcome,
    ScreenTask,
    SupportedClaim,
)

EVIDENCE_EXCERPT_CHARS = 300
RELEVANCE_THRESHOLD = 4

FACET_QUESTIONS = (
    ("research_problem", "What research problem does this paper address, and why?"),
    ("method", "What method, model, or algorithm do the authors propose?"),
    ("experimental_setup", "What datasets, baselines, and experimental setup are used?"),
    ("main_findings", "What are the main quantitative results and findings?"),
    ("limitations", "What limitations, failure cases, or future work do the authors state?"),
)


def screen_candidate(task: ScreenTask) -> ScreenOutcome:
    """Score one candidate against the user query using metadata only."""

    paper = task["paper"]
    prompt = (
        "Decide whether this arXiv paper is relevant to the user's research query.\n"
        "Judge only from the title, authors, date, and abstract below.\n"
        "Use this score rubric:\n"
        "1 = unrelated.\n"
        "2 = weakly related.\n"
        "3 = possibly useful but not central.\n"
        "4 = clearly relevant.\n"
        "5 = highly relevant and should be included unless the paper is low quality.\n"
        "Set is_relevant to true only when score is 4 or 5.\n\n"
        f"User query: {task['user_query']}\n\n"
        f"arXiv ID: {paper.arxiv_id}\n"
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors)}\n"
        f"Published: {paper.published}\n"
        f"Abstract: {paper.abstract}"
    )
    decision = generate_structured(prompt, RelevanceDecision)

    return ScreenOutcome(
        arxiv_id=paper.arxiv_id,
        search_position=task["search_position"],
        score=decision.score,
        reason=decision.reason,
        status="ok",
    )


def screen_candidate_node(task: ScreenTask) -> ReviewerState:
    """Screen one candidate, recording failure instead of aborting the run."""

    try:
        evaluation = with_retries(lambda: screen_candidate(task))
    except Exception as error:
        evaluation = ScreenOutcome(
            arxiv_id=task["paper"].arxiv_id,
            search_position=task["search_position"],
            status="failed",
            error=describe(error),
        )

    return {"candidate_evaluations": [evaluation]}


def select_papers_node(state: ReviewerState) -> ReviewerState:
    """Rank screened candidates deterministically and select the best."""

    evaluations = [
        evaluation
        for evaluation in state.get("candidate_evaluations", [])
        if evaluation.status == "ok" and evaluation.score >= RELEVANCE_THRESHOLD
    ]
    evaluations.sort(key=lambda item: (-item.score, item.search_position))
    target = state.get("target_papers", 4)

    return {"selected_ids": [item.arxiv_id for item in evaluations[:target]]}


def normalize(text: str) -> str:
    """Collapse whitespace and case so excerpts can be matched reliably."""

    return " ".join(text.split()).lower()


def format_context(documents: list[Document]) -> str:
    """Render retrieved chunks so the model can cite them by identifier."""

    blocks = []
    for document in documents:
        metadata = document.metadata
        blocks.append(
            f"[{metadata['chunk_id']}] (page {metadata['page_number']})\n"
            f"{document.page_content}"
        )
    return "\n\n".join(blocks)


def validate_claim(
    claim: DraftClaim,
    arxiv_id: str,
    chunks_by_id: dict[str, Document],
) -> tuple[SupportedClaim | None, int]:
    """Keep only citations that resolve to a shown chunk of the right paper."""

    evidence: list[EvidenceRef] = []
    dropped = 0

    for candidate in claim.evidence:
        document = chunks_by_id.get(candidate.chunk_id)
        excerpt = candidate.excerpt.strip()[:EVIDENCE_EXCERPT_CHARS]

        if (
            document is None
            or document.metadata["arxiv_id"] != arxiv_id
            or not excerpt
            or normalize(excerpt) not in normalize(document.page_content)
        ):
            dropped += 1
            continue

        evidence.append(
            EvidenceRef(
                chunk_id=candidate.chunk_id,
                arxiv_id=arxiv_id,
                page_number=document.metadata["page_number"],
                excerpt=excerpt,
            )
        )

    if not evidence:
        return None, dropped
    return SupportedClaim(text=claim.text.strip(), evidence=evidence), dropped


def analyze_facet(
    task: AnalyzeTask,
    question: str,
) -> tuple[list[SupportedClaim], int, int]:
    """Retrieve evidence for one facet and keep only validated claims."""

    arxiv_id = task["paper"].arxiv_id
    retriever = get_retriever(
        task["thread_id"],
        kind=task.get("retriever_kind", "hybrid-rerank"),
        k=task.get("top_k", DEFAULT_TOP_K),
        fetch_k=task.get("fetch_k", DEFAULT_FETCH_K),
        arxiv_id=arxiv_id,
        multi_query=task.get("multi_query", False),
        data_dir=Path(task["data_dir"]),
    )
    documents = retriever.invoke(question)
    if not documents:
        return [], 0, 0

    chunks_by_id = {
        document.metadata["chunk_id"]: document for document in documents
    }

    prompt = (
        "Answer the question about this paper using only the numbered excerpts below.\n"
        "Write at most 4 short, specific claims. Do not speculate.\n"
        "Every claim must cite at least one excerpt.\n"
        "For each citation, give the chunk_id exactly as shown in square brackets, and "
        "an excerpt copied verbatim from that chunk (at most "
        f"{EVIDENCE_EXCERPT_CHARS} characters).\n"
        "Never invent a chunk_id and never paraphrase inside an excerpt.\n"
        "If the excerpts do not answer the question, return no claims.\n\n"
        f"Question: {question}\n\n"
        f"Excerpts:\n{format_context(documents)}"
    )
    draft = generate_structured(prompt, FacetDraft)

    claims: list[SupportedClaim] = []
    dropped_claims = 0
    dropped_evidence = 0

    for candidate in draft.claims:
        claim, dropped = validate_claim(candidate, arxiv_id, chunks_by_id)
        dropped_evidence += dropped
        if claim is None:
            dropped_claims += 1
            continue
        claims.append(claim)

    return claims, dropped_claims, dropped_evidence


def analyze_paper(task: AnalyzeTask) -> AnalysisOutcome:
    """Download, index, and build a citation-checked analysis of one paper."""

    paper = task["paper"]
    parsed_paper = fetch_parsed_paper(paper)
    documents = chunk_pages(paper.arxiv_id, parsed_paper.pages)
    chunk_count = index_paper(
        task["thread_id"], documents, data_dir=Path(task["data_dir"])
    )

    claims: dict[str, list[SupportedClaim]] = {}
    dropped_claims = 0
    dropped_evidence = 0

    questions = list(FACET_QUESTIONS) + [("relevance_to_query", task["user_query"])]
    for facet, question in questions:
        facet_claims, facet_dropped, evidence_dropped = analyze_facet(task, question)
        dropped_claims += facet_dropped
        dropped_evidence += evidence_dropped
        if facet_claims:
            claims[facet] = facet_claims

    return AnalysisOutcome(
        arxiv_id=paper.arxiv_id,
        search_position=task["search_position"],
        status="ok",
        chunk_count=chunk_count,
        analysis=GroundedAnalysis(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            claims=claims,
            dropped_claims=dropped_claims,
            dropped_evidence=dropped_evidence,
        ),
    )


def analyze_paper_node(task: AnalyzeTask) -> ReviewerState:
    """Analyze one paper, recording failure instead of aborting the run."""

    try:
        outcome = with_retries(lambda: analyze_paper(task))
    except Exception as error:
        outcome = AnalysisOutcome(
            arxiv_id=task["paper"].arxiv_id,
            search_position=task["search_position"],
            status="failed",
            error=describe(error),
        )

    return {"analysis_outcomes": [outcome]}
