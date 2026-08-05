#!/usr/bin/env python3
from __future__ import annotations

import arxiv
import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, TypeVar, TypedDict

import fitz
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


GEMINI_MODEL = "gemini-3.1-flash-lite"
RELEVANCE_TEXT_CHARS = 12000
T = TypeVar("T", bound=BaseModel)


# SearchPlan stores the arXiv search queries generated from the user query.
class SearchPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=3)


# PaperMetadata stores normalized arXiv metadata for one candidate paper.
class PaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_url: str
    entry_url: str


# ParsedPaper stores the extracted text and page count for one PDF.
class ParsedPaper(BaseModel):
    arxiv_id: str
    text: str
    page_count: int = Field(ge=1)


# RelevanceDecision stores the model's relevance score and explanation.
class RelevanceDecision(BaseModel):
    arxiv_id: str
    is_relevant: bool
    score: int = Field(ge=1, le=5)
    reason: str


# PaperAnalysis stores the detailed notes extracted from one relevant paper.
class PaperAnalysis(BaseModel):
    arxiv_id: str
    title: str
    research_problem: str
    method: str
    experimental_setup: str
    main_findings: str
    limitations: str
    relevance_to_query: str


# LiteratureReview stores the structured content for a complete review.
class LiteratureReview(BaseModel):
    title: str
    overview: str
    papers: list[PaperAnalysis]
    themes: list[str]
    gaps: list[str]
    suggested_reading_order: list[str]


# ReviewerState stores the shared fields that graph nodes read and update.
class ReviewerState(TypedDict, total=False):
    user_query: str
    max_results: int
    target_papers: int
    output: Path
    checkpoint: Path
    search_queries: list[str]
    found_papers: list[PaperMetadata]
    current_paper_index: int
    parsed_papers: dict[str, ParsedPaper]
    relevance_decisions: dict[str, RelevanceDecision]
    chosen_papers: dict[str, PaperAnalysis]
    markdown: str


# gemini_client creates a Gemini client from the configured API key.
def gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
    return genai.Client(api_key=api_key)


# generate_text sends a plain text prompt to Gemini and returns text.
def generate_text(prompt: str) -> str:
    client = gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text or ""


# generate_structured sends a prompt to Gemini and validates a Pydantic result.
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


# make_search_plan converts the user query into arXiv search queries.
def make_search_plan(user_query: str) -> SearchPlan:
    prompt = (
        "Convert this research question into 1 to 3 concise arXiv search queries. "
        "Use technical keywords that are likely to appear in paper titles or abstracts. "
        "Keep each query focused on the user's research domain. "
        "For language-model questions, each query should include LLM, language model, "
        "AI alignment, or AI agent terminology. "
        "Do not write explanations.\n\n"
        f"Research question: {user_query}"
    )
    return generate_structured(prompt, SearchPlan)


# search_arxiv runs one arXiv query and returns normalized paper metadata.
def search_arxiv(arxiv_query: str, max_results: int) -> list[PaperMetadata]:

    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=arxiv_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[PaperMetadata] = []
    seen_ids: set[str] = set()

    # Iterate through arXiv results and normalize each unseen paper.
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


# search_node generates arXiv queries, searches arXiv, and deduplicates papers.
def search_node(state: ReviewerState) -> ReviewerState:
    user_query = state["user_query"]
    max_results = state.get("max_results", 10)
    search_plan = make_search_plan(user_query)
    arxiv_queries = [query.strip() for query in search_plan.queries if query.strip()]

    if not arxiv_queries:
        arxiv_queries = [user_query]

    found_papers: list[PaperMetadata] = []
    seen_ids: set[str] = set()

    # Run each generated arXiv query until the overall result limit is reached.
    for index, arxiv_query in enumerate(arxiv_queries):
        if len(found_papers) >= max_results:
            break
        if index > 0:
            time.sleep(3.0)

        remaining = max_results - len(found_papers)
        remaining_queries = len(arxiv_queries) - index
        query_limit = max(1, (remaining + remaining_queries - 1) // remaining_queries)
        # Add papers from this query while removing duplicates across queries.
        for paper in search_arxiv(arxiv_query, query_limit):
            if paper.arxiv_id in seen_ids:
                continue
            seen_ids.add(paper.arxiv_id)
            found_papers.append(paper)
            if len(found_papers) >= max_results:
                break

    return {
        "search_queries": arxiv_queries,
        "found_papers": found_papers,
    }


# route_after_search chooses parsing or final writing after arXiv search.
def route_after_search(state: ReviewerState) -> str:
    if state.get("found_papers"):
        return "download_parse"
    return "write_markdown"


# download_parse_node downloads the current paper PDF and extracts its text.
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


# relevance_eval_node scores the current paper against the user query.
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


# route_after_relevance_eval chooses extraction or paper advancement.
def route_after_relevance_eval(state: ReviewerState) -> str:
    current_paper_index = state.get("current_paper_index", 0)
    paper = state["found_papers"][current_paper_index]
    decision = state["relevance_decisions"][paper.arxiv_id]

    if decision.is_relevant:
        return "extract_core"
    return "advance_paper"


# advance_paper_node moves the graph state to the next candidate paper.
def advance_paper_node(state: ReviewerState) -> ReviewerState:
    return {"current_paper_index": state.get("current_paper_index", 0) + 1}


# extract_core_node extracts structured notes from the current relevant paper.
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


# route_after_extract_core chooses whether to continue or write the review.
def route_after_extract_core(state: ReviewerState) -> str:
    chosen_papers = state.get("chosen_papers", {})
    target_papers = state.get("target_papers", 4)

    if len(chosen_papers) >= target_papers:
        return "write_markdown"
    return "advance_paper"


# route_after_advance_paper chooses the next node after advancing papers.
def route_after_advance_paper(state: ReviewerState) -> str:
    current_paper_index = state.get("current_paper_index", 0)

    if current_paper_index < len(state.get("found_papers", [])):
        return "download_parse"
    return "write_markdown"


# render_markdown_fallback renders selected paper analyses without an LLM call.
def render_markdown_fallback(state: ReviewerState) -> str:
    user_query = state["user_query"]
    search_queries = state.get("search_queries", [])
    found_papers = state.get("found_papers", [])
    chosen_papers = state.get("chosen_papers", {})

    metadata_by_id = {paper.arxiv_id: paper for paper in found_papers}
    analyses = list(chosen_papers.values())

    lines = [
        f"# Literature Review: {user_query}",
        "",
        "## Search Summary",
        "",
        f"- User query: {user_query}",
        f"- arXiv queries: {', '.join(search_queries) if search_queries else 'None recorded'}",
        f"- Candidate papers found: {len(found_papers)}",
        f"- Relevant papers selected: {len(analyses)}",
        "",
        "## Overview",
        "",
    ]

    if analyses:
        lines.append(
            f"This review summarizes {len(analyses)} paper"
            f"{'' if len(analyses) == 1 else 's'} selected as relevant to the query."
        )
    else:
        lines.append("No relevant papers were selected.")

    lines.extend(["", "## Paper Notes", ""])

    # Render a detailed notes section for each selected paper.
    for analysis in analyses:
        paper = metadata_by_id.get(analysis.arxiv_id)
        authors = ", ".join(paper.authors) if paper else "Unknown authors"
        published = paper.published if paper else "Unknown date"
        entry_url = paper.entry_url if paper else ""

        title = f"[{analysis.title}]({entry_url})" if entry_url else analysis.title
        lines.extend(
            [
                f"### {title}",
                "",
                f"- arXiv ID: {analysis.arxiv_id}",
                f"- Authors: {authors}",
                f"- Published: {published}",
                f"- Research problem: {analysis.research_problem}",
                f"- Method: {analysis.method}",
                f"- Experimental setup: {analysis.experimental_setup}",
                f"- Main findings: {analysis.main_findings}",
                f"- Limitations: {analysis.limitations}",
                f"- Relevance to query: {analysis.relevance_to_query}",
                "",
            ]
        )

    lines.extend(
        [
            "## Comparison Table",
            "",
            "| Paper | Method | Main findings | Limitations |",
            "| --- | --- | --- | --- |",
        ]
    )

    # Render one comparison-table row for each selected paper.
    for analysis in analyses:
        title = analysis.title.replace("|", "\\|").replace("\n", " ")
        method = analysis.method.replace("|", "\\|").replace("\n", " ")
        findings = analysis.main_findings.replace("|", "\\|").replace("\n", " ")
        limitations = analysis.limitations.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {title} | {method} | {findings} | {limitations} |")

    lines.extend(["", "## Research Themes", ""])
    if analyses:
        # Summarize why each selected paper matters for the user query.
        for analysis in analyses:
            lines.append(f"- {analysis.title}: {analysis.relevance_to_query}")
    else:
        lines.append("No selected papers are available to summarize.")

    lines.extend(["", "## Research Gaps", ""])
    if analyses:
        # Reuse each paper's stated limitations as initial research gaps.
        for analysis in analyses:
            lines.append(f"- {analysis.title}: {analysis.limitations}")
    else:
        lines.append("No selected papers are available to summarize.")

    lines.extend(["", "## Suggested Reading Order", ""])
    if analyses:
        # List selected papers in the order they were accepted.
        for index, analysis in enumerate(analyses, start=1):
            lines.append(f"{index}. {analysis.title}")
    else:
        lines.append("No selected papers are available to order.")

    markdown = "\n".join(lines).rstrip() + "\n"
    return markdown


# write_markdown_node asks Gemini to write the final Markdown review.
def write_markdown_node(state: ReviewerState) -> ReviewerState:
    output = state.get("output", Path("review.md"))
    chosen_papers = state.get("chosen_papers", {})

    if not chosen_papers:
        markdown = render_markdown_fallback(state)
        output.write_text(markdown, encoding="utf-8")
        return {"markdown": markdown}

    papers_for_prompt = []
    metadata_by_id = {paper.arxiv_id: paper for paper in state.get("found_papers", [])}

    # Build compact paper records for the final review prompt.
    for analysis in chosen_papers.values():
        paper = metadata_by_id.get(analysis.arxiv_id)
        papers_for_prompt.append(
            {
                "metadata": paper.model_dump() if paper else {},
                "analysis": analysis.model_dump(),
            }
        )

    prompt = (
        "Write a polished Markdown literature review from these structured paper notes. "
        "Use only the facts provided here. Do not invent papers, claims, results, or citations. "
        "Output Markdown only, with no code fences.\n\n"
        "Required sections:\n"
        "# Literature Review: <user query>\n"
        "## Search Summary\n"
        "## Overview\n"
        "## Key Papers\n"
        "## Comparison Table\n"
        "## Research Themes\n"
        "## Research Gaps\n"
        "## Suggested Reading Order\n\n"
        f"User query: {state['user_query']}\n"
        f"arXiv search queries: {state.get('search_queries', [])}\n"
        f"Candidate papers found: {len(state.get('found_papers', []))}\n"
        f"Selected paper notes:\n{json.dumps(papers_for_prompt, indent=2)}"
    )

    try:
        markdown = generate_text(prompt).strip()
    except Exception:
        markdown = ""

    if not markdown:
        markdown = render_markdown_fallback(state)
    else:
        markdown = markdown.rstrip() + "\n"

    output.write_text(markdown, encoding="utf-8")
    return {"markdown": markdown}


# checkpoint_path returns the JSON checkpoint path for the current state.
def checkpoint_path(state: ReviewerState) -> Path:
    if "checkpoint" in state:
        return state["checkpoint"]

    output = state.get("output", Path("review.md"))
    if output.suffix:
        return output.with_suffix(f"{output.suffix}.checkpoint.json")
    return output.with_name(f"{output.name}.checkpoint.json")


# to_jsonable converts state values into JSON-compatible values.
def to_jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


# save_checkpoint_node writes the current graph state to a JSON file.
def save_checkpoint_node(state: ReviewerState) -> ReviewerState:
    path = checkpoint_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(state), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {"checkpoint": path}


# checkpointed_node saves graph state after a regular graph node finishes.
def checkpointed_node(
    node_function: Callable[[ReviewerState], ReviewerState],
) -> Callable[[ReviewerState], ReviewerState]:
    # wrapped_node merges a node update into state before checkpointing.
    def wrapped_node(state: ReviewerState) -> ReviewerState:
        update = node_function(state)
        merged_state = dict(state)
        merged_state.update(update)
        save_checkpoint_node(merged_state)
        return update

    return wrapped_node


# build_graph assembles the LangGraph workflow from the node functions.
def build_graph():
    graph = StateGraph(ReviewerState)
    graph.add_node("search", checkpointed_node(search_node))
    graph.add_node("download_parse", checkpointed_node(download_parse_node))
    graph.add_node("relevance_eval", checkpointed_node(relevance_eval_node))
    graph.add_node("advance_paper", checkpointed_node(advance_paper_node))
    graph.add_node("extract_core", checkpointed_node(extract_core_node))
    graph.add_node("write_markdown", checkpointed_node(write_markdown_node))

    graph.add_edge(START, "search")
    graph.add_conditional_edges(
        "search",
        route_after_search,
        {
            "download_parse": "download_parse",
            "write_markdown": "write_markdown",
        },
    )
    graph.add_edge("download_parse", "relevance_eval")
    graph.add_conditional_edges(
        "relevance_eval",
        route_after_relevance_eval,
        {
            "extract_core": "extract_core",
            "advance_paper": "advance_paper",
        },
    )
    graph.add_conditional_edges(
        "extract_core",
        route_after_extract_core,
        {
            "advance_paper": "advance_paper",
            "write_markdown": "write_markdown",
        },
    )
    graph.add_conditional_edges(
        "advance_paper",
        route_after_advance_paper,
        {
            "download_parse": "download_parse",
            "write_markdown": "write_markdown",
        },
    )
    graph.add_edge("write_markdown", END)
    return graph.compile()


# run_reviewer invokes the compiled graph with initial user settings.
def run_reviewer(
    user_query: str,
    max_results: int,
    target_papers: int,
    output: Path,
    checkpoint: Path | None,
) -> ReviewerState:
    state: ReviewerState = {
        "user_query": user_query,
        "max_results": max_results,
        "target_papers": target_papers,
        "output": output,
        "current_paper_index": 0,
        "parsed_papers": {},
        "relevance_decisions": {},
        "chosen_papers": {},
    }
    if checkpoint:
        state["checkpoint"] = checkpoint

    return build_graph().invoke(state)


# main validates command-line arguments and required environment variables.
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an arXiv literature reviewer and write a Markdown report."
    )
    parser.add_argument(
        "--user-query",
        required=True,
        help="Research question or topic to review.",
    )
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--target-papers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("review.md"))
    parser.add_argument("--checkpoint", type=Path)
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

    run_reviewer(
        user_query=args.user_query,
        max_results=args.max_results,
        target_papers=args.target_papers,
        output=args.output,
        checkpoint=args.checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
