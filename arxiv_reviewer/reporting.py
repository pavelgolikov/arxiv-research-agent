"""Markdown report rendering and synthesis."""

import json
from pathlib import Path

from .gemini_client import generate_text
from .review_types import ReviewerState


def render_markdown_fallback(state: ReviewerState) -> str:
    """Render selected paper analyses without an LLM call."""

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

    for analysis in analyses:
        title = analysis.title.replace("|", "\\|").replace("\n", " ")
        method = analysis.method.replace("|", "\\|").replace("\n", " ")
        findings = analysis.main_findings.replace("|", "\\|").replace("\n", " ")
        limitations = analysis.limitations.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {title} | {method} | {findings} | {limitations} |")

    lines.extend(["", "## Research Themes", ""])
    if analyses:
        for analysis in analyses:
            lines.append(f"- {analysis.title}: {analysis.relevance_to_query}")
    else:
        lines.append("No selected papers are available to summarize.")

    lines.extend(["", "## Research Gaps", ""])
    if analyses:
        for analysis in analyses:
            lines.append(f"- {analysis.title}: {analysis.limitations}")
    else:
        lines.append("No selected papers are available to summarize.")

    lines.extend(["", "## Suggested Reading Order", ""])
    if analyses:
        for index, analysis in enumerate(analyses, start=1):
            lines.append(f"{index}. {analysis.title}")
    else:
        lines.append("No selected papers are available to order.")

    return "\n".join(lines).rstrip() + "\n"


def write_atomically(output: Path, markdown: str) -> None:
    """Write the report through a temporary file and replace the target."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(markdown, encoding="utf-8")
    temporary.replace(output)


def write_markdown_node(state: ReviewerState) -> ReviewerState:
    """Ask Gemini to write the final Markdown review."""

    output = Path(state.get("output", "review.md"))
    chosen_papers = state.get("chosen_papers", {})

    if not chosen_papers:
        markdown = render_markdown_fallback(state)
        write_atomically(output, markdown)
        return {"markdown": markdown, "status": "empty"}

    papers_for_prompt = []
    metadata_by_id = {paper.arxiv_id: paper for paper in state.get("found_papers", [])}

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
        status = "partial"
    else:
        markdown = markdown.rstrip() + "\n"
        status = "complete"

    write_atomically(output, markdown)
    return {"markdown": markdown, "status": status}
