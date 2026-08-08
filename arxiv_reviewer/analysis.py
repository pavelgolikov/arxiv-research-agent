"""Paper relevance and grounded per-facet analysis graph nodes."""

from langchain_core.documents import Document

from .gemini_client import generate_structured
from .rag import DEFAULT_FETCH_K, DEFAULT_TOP_K, data_dir_path, get_retriever
from .review_types import (
    DraftClaim,
    EvidenceRef,
    FacetDraft,
    GroundedAnalysis,
    RelevanceDecision,
    ReviewerState,
    SupportedClaim,
)

RELEVANCE_TEXT_CHARS = 12000
EVIDENCE_EXCERPT_CHARS = 300

FACET_QUESTIONS = (
    ("research_problem", "What research problem does this paper address, and why?"),
    ("method", "What method, model, or algorithm do the authors propose?"),
    ("experimental_setup", "What datasets, baselines, and experimental setup are used?"),
    ("main_findings", "What are the main quantitative results and findings?"),
    ("limitations", "What limitations, failure cases, or future work do the authors state?"),
)


def relevance_eval_node(state: ReviewerState) -> ReviewerState:
    """Score the current paper against the user query."""

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
        f"Paper text preview:\n{parsed_paper.full_text[:RELEVANCE_TEXT_CHARS]}"
    )
    decision = generate_structured(prompt, RelevanceDecision)
    decision = decision.model_copydroppe(
        update={"arxiv_id": paper.arxiv_id, "is_relevant": decision.score >= 4}
    )
    relevance_decisions[paper.arxiv_id] = decision

    return {"relevance_decisions": relevance_decisions}


def route_after_relevance_eval(state: ReviewerState) -> str:
    """Choose extraction or paper advancement after relevance evaluation."""

    current_paper_index = state.get("current_paper_index", 0)
    paper = state["found_papers"][current_paper_index]
    decision = state["relevance_decisions"][paper.arxiv_id]

    if decision.is_relevant:
        return "extract_core"
    return "advance_paper"


def advance_paper_node(state: ReviewerState) -> ReviewerState:
    """Move the graph state to the next candidate paper."""

    return {"current_paper_index": state.get("current_paper_index", 0) + 1}


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
    state: ReviewerState,
    arxiv_id: str,
    question: str,
) -> tuple[list[SupportedClaim], int, int]:
    """Retrieve evidence for one facet and keep only validated claims."""

    retriever = get_retriever(
        state["thread_id"],
        kind=state.get("retriever_kind", "hybrid-rerank"),
        k=state.get("top_k", DEFAULT_TOP_K),
        fetch_k=state.get("fetch_k", DEFAULT_FETCH_K),
        arxiv_id=arxiv_id,
        multi_query=state.get("multi_query", False),
        data_dir=data_dir_path(state),
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


def extract_core_node(state: ReviewerState) -> ReviewerState:
    """Build a grounded, citation-checked analysis of the current paper."""

    current_paper_index = state.get("current_paper_index", 0)
    paper = state["found_papers"][current_paper_index]
    chosen_papers = dict(state.get("chosen_papers", {}))

    if paper.arxiv_id in chosen_papers:
        return {"chosen_papers": chosen_papers}

    claims: dict[str, list[SupportedClaim]] = {}
    dropped_claims = 0
    dropped_evidence = 0

    questions = list(FACET_QUESTIONS) + [("relevance_to_query", state["user_query"])]
    for facet, question in questions:
        facet_claims, facet_dropped, evidence_dropped = analyze_facet(
            state, paper.arxiv_id, question
        )
        dropped_claims += facet_dropped
        dropped_evidence += evidence_dropped
        if facet_claims:
            claims[facet] = facet_claims

    chosen_papers[paper.arxiv_id] = GroundedAnalysis(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        claims=claims,
        dropped_claims=dropped_claims,
        dropped_evidence=dropped_evidence,
    )

    return {"chosen_papers": chosen_papers}


def route_after_extract_core(state: ReviewerState) -> str:
    """Choose whether to continue or write the review."""

    chosen_papers = state.get("chosen_papers", {})
    target_papers = state.get("target_papers", 4)

    if len(chosen_papers) >= target_papers:
        return "write_markdown"
    return "advance_paper"


def route_after_advance_paper(state: ReviewerState) -> str:
    """Choose the next node after advancing papers."""

    current_paper_index = state.get("current_paper_index", 0)

    if current_paper_index < len(state.get("found_papers", [])):
        return "download_parse"
    return "write_markdown"
