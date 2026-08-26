"""Candidate screening and grounded per-facet analysis branches."""

import unicodedata
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
    SupportVerdicts,
)

EVIDENCE_EXCERPT_CHARS = 300

# Chosen by sweeping this value through `select_papers_node` against the labeled
# screening set, not by guessing: see `evals/results/screening.json`. Thresholds 3 and
# 4 score identical precision, but 3 recovers more central papers (0.683 against 0.611)
# and leaves fewer reviews short of their target. Below 3 the model starts admitting
# papers it scored "weakly related", and precision falls.
RELEVANCE_THRESHOLD = 3

# Citations the judge grades 0 are discarded; grade 1 survives. The hand-labeled
# sample behind `evals/labels/claim_support_labels.json` is the reason: 0 of 40
# citations failed outright, and all 9 partial grades were one facet's enumerations —
# a claim listing five datasets, citing the fragment that names two. Dropping those
# would discard mostly-correct work to fix a claim-phrasing problem. Kept as a
# constant so `evals.run_claim_judge` can sweep it against the labels rather than
# against a copy of this rule.
SUPPORT_THRESHOLD = 1

# The judge and the hand-labeling sheet have to grade the same question, or scoring
# one against the other measures nothing. Both render this text: `judge_support` puts
# it in the prompt, `evals.build.claim_support.write_sheet` puts it in the sheet a
# human fills in.
SUPPORT_RUBRIC = (
    "- `2` — the excerpt establishes the claim.\n"
    "- `1` — the excerpt supports part of the claim, or supports it with a qualifier "
    "the quote does not carry.\n"
    "- `0` — the excerpt does not support the claim.\n"
    "\n"
    "Judge the **excerpt against the claim**, not whether the claim is true of the "
    "paper. A correct statement quoted from the wrong sentence is still a `0`."
)

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

    # Read from state so the screening evaluation can sweep the threshold through the
    # real selection rule instead of a copy of it that could drift. Unset means the
    # module default, so production behaviour is unchanged.
    threshold = state.get("relevance_threshold", RELEVANCE_THRESHOLD)
    evaluations = [
        evaluation
        for evaluation in state.get("candidate_evaluations", [])
        if evaluation.status == "ok" and evaluation.score >= threshold
    ]
    evaluations.sort(key=lambda item: (-item.score, item.search_position))
    target = state.get("target_papers", 4)

    return {"selected_ids": [item.arxiv_id for item in evaluations[:target]]}


def normalize(text: str) -> str:
    """Fold whitespace, case, unicode form, and hyphenation for excerpt matching.

    PDF extraction preserves the hyphens a typesetter inserted at line breaks, so a
    chunk can read "lead- ing" where the paper reads "leading". A model quoting that
    passage faithfully writes "leading" and the citation was then rejected, which
    measured the PDF extractor rather than the model: on two sample papers this
    discarded 63% of citations that were in fact verbatim.

    Hyphens are removed from both sides, so the comparison no longer depends on where
    a line happened to break. Order matters — whitespace is collapsed first, so that
    "lead- ing" and "fine-tuning" both fold to their unhyphenated form. Matching a
    300-character excerpt as a substring remains a strong constraint, and fabricated
    or paraphrased text still fails.
    """

    collapsed = " ".join(unicodedata.normalize("NFKC", text).split()).lower()
    return collapsed.replace("- ", "").replace("-", "")


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


def judge_support(items: list[tuple[str, str]]) -> list[int | None]:
    """Grade how well each excerpt supports the claim it was cited for.

    One call covers a whole facet — at most four claims and their citations — so the
    check costs one model call per facet rather than one per citation.

    Returns one grade per item, in the order given. A verdict the model omits, or
    indexes to something that was never sent, comes back as `None`. `apply_support_judge`
    treats that as a failed check rather than a pass: a citation must not reach the
    report because a reply had a hole in it.
    """

    numbered = "\n\n".join(
        f"{position}.\nClaim: {claim}\nExcerpt: {excerpt}"
        for position, (claim, excerpt) in enumerate(items, start=1)
    )
    prompt = (
        "Grade how well each excerpt below supports the claim it was cited for.\n\n"
        f"{SUPPORT_RUBRIC}\n\n"
        "Return one verdict per item, using the item's own number as its index, and "
        "give a short reason for each.\n\n"
        f"{numbered}"
    )
    graded = generate_structured(prompt, SupportVerdicts)

    by_index = {verdict.index: verdict.grade for verdict in graded.verdicts}
    return [by_index.get(position) for position in range(1, len(items) + 1)]


def apply_support_judge(
    claims: list[SupportedClaim],
) -> tuple[list[SupportedClaim], int, int]:
    """Drop citations whose excerpt does not support the claim it was cited for.

    Runs after `validate_claim`, so every quote reaching the judge is already proven
    to occur in the chunk it cites. The question left is the one no deterministic
    check can answer: does the quote support the sentence built on it.
    """

    pairs = [
        (claim.text, evidence.excerpt)
        for claim in claims
        for evidence in claim.evidence
    ]
    if not pairs:
        return [], 0, 0

    grades = judge_support(pairs)

    supported: list[SupportedClaim] = []
    dropped_claims = 0
    dropped_unsupported = 0
    position = 0

    for claim in claims:
        kept: list[EvidenceRef] = []
        for evidence in claim.evidence:
            grade = grades[position]
            position += 1

            if grade is None or grade < SUPPORT_THRESHOLD:
                dropped_unsupported += 1
                continue

            kept.append(evidence.model_copy(update={"support_grade": grade}))

        if not kept:
            dropped_claims += 1
            continue
        supported.append(claim.model_copy(update={"evidence": kept}))

    return supported, dropped_claims, dropped_unsupported


def analyze_facet(
    task: AnalyzeTask,
    question: str,
) -> tuple[list[SupportedClaim], int, int, int]:
    """Retrieve evidence for one facet and keep only validated, supported claims."""

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
        return [], 0, 0, 0

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

    supported, unsupported_claims, dropped_unsupported = apply_support_judge(claims)

    return (
        supported,
        dropped_claims + unsupported_claims,
        dropped_evidence,
        dropped_unsupported,
    )


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
    dropped_unsupported = 0

    questions = list(FACET_QUESTIONS) + [("relevance_to_query", task["user_query"])]
    for facet, question in questions:
        facet_claims, facet_dropped, evidence_dropped, unsupported = analyze_facet(
            task, question
        )
        dropped_claims += facet_dropped
        dropped_evidence += evidence_dropped
        dropped_unsupported += unsupported
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
            dropped_unsupported=dropped_unsupported,
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
