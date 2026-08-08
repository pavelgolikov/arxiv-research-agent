"""Markdown report rendering and synthesis."""

import json
from pathlib import Path

from .gemini_client import generate_text
from .review_types import GroundedAnalysis, ReviewerState, SupportedClaim

FACET_TITLES = {
    "research_problem": "Research problem",
    "method": "Method",
    "experimental_setup": "Experimental setup",
    "main_findings": "Main findings",
    "limitations": "Limitations",
    "relevance_to_query": "Relevance to the query",
}


def escape_cell(text: str) -> str:
    """Make text safe to place inside a Markdown table cell."""

    return text.replace("|", "\\|").replace("\n", " ")


def citation_link(arxiv_id: str, page_number: int) -> str:
    """Build a page-anchored link to the cited PDF."""

    return f"[p. {page_number}](https://arxiv.org/pdf/{arxiv_id}#page={page_number})"


def render_claim(claim: SupportedClaim) -> str:
    """Render one claim followed by its verified citations."""

    citations = " ".join(
        citation_link(evidence.arxiv_id, evidence.page_number)
        for evidence in claim.evidence
    )
    return f"{claim.text.rstrip('.')}. {citations}"


def claim_texts(analysis: GroundedAnalysis, facet: str) -> list[str]:
    """Return the plain claim sentences recorded for one facet."""

    return [claim.text for claim in analysis.claims.get(facet, [])]


def first_claim_text(analysis: GroundedAnalysis, facet: str) -> str:
    """Return one representative claim for compact table cells."""

    texts = claim_texts(analysis, facet)
    return texts[0] if texts else "Not reported"


def render_markdown_fallback(state: ReviewerState) -> str:
    """Render validated paper analyses without an LLM call."""

    user_query = state["user_query"]
    search_queries = state.get("search_queries", [])
    found_papers = state.get("found_papers}", [])
    chosen_papers = state.get("chosen_papers", {})

    metadata_by_id = {paper.arxiv_id: paper for paper in found_papers}
    analyses = list(chosen_papers.values())
    total_claims = sum(analysis.supported_claim_count for analysis in analyses)
    total_dropped = sum(analysis.dropped_claims for analysis in analyses)

    lines = [
        f"# Literature Review: {user_query}",
        "",
        "## Search Summary",
        "",
        f"- User query: {user_query}",
        f"- arXiv queries: {', '.join(search_queries) if search_queries else 'None recorded'}",
        f"- Candidate papers found: {len(found_papers)}",
        f"- Relevant papers selected: {len(analyses)}",
        f"- Retrieval strategy: {state.get('retriever_kind', 'dense')}",
        f"- Supported claims: {total_claims}",
        f"- Claims dropped in citation validation: {total_dropped}",
        "",
        "## Method and Limitations Notice",
        "",
        "Every claim below was generated from retrieved excerpts of the cited paper "
        "and kept only when its citation resolved to a real chunk of that paper and "
        "its quoted excerpt was found in that chunk. Claims failing this check were "
        "discarded rather than reported.",
        "",
        "## Overview",
        "",
    ]

    if analyses:
        lines.append(
            f"This review summarizes {len(analyses)} paper"
            f"{'' if len(analyses) == 1 else 's'} selected as relevant to the query, "
            f"supported by {total_claims} cited claims."
        )
    else:
        lines.append("No relevant papers were selected.")

    lines.extend(["", "## Key Papers", ""])

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
                "",
            ]
        )

        for facet, heading in FACET_TITLES.items():
            claims = analysis.claims.get(facet, [])
            if not claims:
                continue
            lines.append(f"**{heading}**")
            lines.append("")
            lines.extend(f"- {render_claim(claim)}" for claim in claims)
            lines.append("")

        if analysis.is_partial:
            lines.extend(
                [
                    f"_Citation validation dropped {analysis.dropped_claims} claim(s) "
                    f"and {analysis.dropped_evidence} citation(s) for this paper._",
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
        lines.append(
            f"| {escape_cell(analysis.title)} "
            f"| {escape_cell(first_claim_text(analysis, 'method'))} "
            f"| {escape_cell(first_claim_text(analysis, 'main_findings'))} "
            f"| {escape_cell(first_claim_text(analysis, 'limitations'))} |"
        )

    lines.extend(["", "## Research Themes", ""])
    if analyses:
        for analysis in analyses:
            lines.append(
                f"- {analysis.title}: "
                f"{first_claim_text(analysis, 'relevance_to_query')}"
            )
    else:
        lines.append("No selected papers are available to summarize.")

    lines.extend(["", "## Research Gaps", ""])
    if analyses:
        for analysis in analyses:
            lines.append(
                f"- {analysis.title}: {first_claim_text(analysis, 'limitations')}"
            )
    else:
        lines.append("No selected papers are available to summarize.")

    lines.extend(["", "## Suggested Reading Order", ""])
    if analyses:
        for index, analysis in enumerate(analyses, start=1):
            lines.append(f"{index}. {analysis.title}")
    else:
        lines.append("No selected papers are available to order.")

    return "\n".join(lines).rstrip() + "\n"


def synthesis_payload(state: ReviewerState) -> list[dict]:
    """Package validated claims and citations for the synthesis prompt."""

    metadata_by_id = {paper.arxiv_id: paper for paper in state.get("found_papers", [])}
    payload = []

    for analysis in state.get("chosen_papers", {}).values():
        paper = metadata_by_id.get(analysis.arxiv_id)
        payload.append(
            {
                "metadata": paper.model_dump() if paper else {},
                "claims": {
                    facet: [
                        {
                            "text": claim.text,
                            "citations": [
                                {
                                    "arxiv_id": evidence.arxiv_id,
                                    "page": evidence.page_number,
                                }
                                for evidence in claim.evidence
                            ],
                        }
                        for claim in claims
                    ]
                    for facet, claims in analysis.claims.items()
                },
            }
        )

    return payload


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

    prompt = (
        "Write a polished Markdown literature review from these validated paper claims. "
        "Use only the claims provided here. Do not invent papers, claims, results, or "
        "citations. Preserve every citation as a Markdown link of the form "
        "[p. N](https://arxiv.org/pdf/ARXIV_ID#page=N) using the arxiv_id and page "
        "recorded with each claim. Output Markdown only, with no code fences.\n\n"
        "Required sections:\n"
        "# Literature Review: <user query>\n"
        "## Search Summary\n"
        "## Method and Limitations Notice\n"
        "## Overview\n"
        "## Key Papers\n"
        "## Comparison Table\n"
        "## Research Themes\n"
        "## Research Gaps\n"
        "## Suggested Reading Order\n\n"
        f"User query: {state['user_query']}\n"
        f"arXiv search queries: {state.get('search_queries', [])}\n"
        f"Candidate papers found: {len(state.get('found_papers', []))}\n"
        f"Retrieval strategy: {state.get('retriever_kind', 'dense')}\n"
        f"Validated paper claims:\n{json.dumps(synthesis_payload(state), indent=2)}"
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
