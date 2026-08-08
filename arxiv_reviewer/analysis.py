"""Paper relevance and structured-analysis graph nodes."""

from .gemini_client import generate_structured
from .review_types import PaperAnalysis, RelevanceDecision, ReviewerState


RELEVANCE_TEXT_CHARS = 12000


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
    decision = decision.model_copy(
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


def extract_core_node(state: ReviewerState) -> ReviewerState:
    """Extract structured notes from the current relevant paper."""

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
        f"Full paper text:\n{parsed_paper.full_text}"
    )
    analysis = generate_structured(prompt, PaperAnalysis)
    analysis = analysis.model_copy(
        update={"arxiv_id": paper.arxiv_id, "title": paper.title}
    )
    chosen_papers[paper.arxiv_id] = analysis

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
