"""arXiv search, PDF download, and parsing graph nodes."""

import time

import arxiv
import httpx
import pymupdf

from .failures import PaperUnusableError
from .gemini_client import generate_structured
from .review_types import (
    PaperMetadata,
    ParsedPage,
    ParsedPaper,
    ReviewerState,
    SearchPlan,
)

MAX_PDF_BYTES = 50 * 1024 * 1024


def make_search_plan(user_query: str) -> SearchPlan:
    """Convert the user query into arXiv search queries."""

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


def search_arxiv(arxiv_query: str, max_results: int) -> list[PaperMetadata]:
    """Run one arXiv query and return normalized paper metadata."""

    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=arxiv_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[PaperMetadata] = []
    seen_ids: set[str] = set()

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


def search_node(state: ReviewerState) -> ReviewerState:
    """Generate arXiv queries, search arXiv, and deduplicate papers."""

    user_query = state["user_query"]
    max_results = state.get("max_results", 10)
    search_plan = make_search_plan(user_query)
    arxiv_queries = [query.strip() for query in search_plan.queries if query.strip()]

    if not arxiv_queries:
        arxiv_queries = [user_query]

    found_papers: list[PaperMetadata] = []
    seen_ids: set[str] = set()

    for index, arxiv_query in enumerate(arxiv_queries):
        if len(found_papers) >= max_results:
            break
        if index > 0:
            time.sleep(3.0)

        remaining = max_results - len(found_papers)
        remaining_queries = len(arxiv_queries) - index
        query_limit = max(1, (remaining + remaining_queries - 1) // remaining_queries)
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


def download_paper(paper: PaperMetadata) -> bytes:
    """Fetch one paper PDF, rejecting responses that are not usable PDFs."""

    response = httpx.get(paper.pdf_url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    content = response.content

    if len(content) > MAX_PDF_BYTES:
        raise PaperUnusableError(
            f"PDF for {paper.arxiv_id} is {len(content)} bytes, "
            f"over the {MAX_PDF_BYTES} byte limit."
        )
    if not content.startswith(b"%PDF"):
        raise PaperUnusableError(f"Response for {paper.arxiv_id} is not a PDF.")

    return content


def parse_paper(paper: PaperMetadata, content: bytes) -> ParsedPaper:
    """Extract text from a downloaded PDF one page at a time."""

    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            pages = [
                ParsedPage(page_number=number, text=page.get_text())
                for number, page in enumerate(document, start=1)
            ]
            page_count = document.page_count
    except PaperUnusableError:
        raise
    except Exception as error:
        raise PaperUnusableError(
            f"Could not parse PDF for {paper.arxiv_id}: {error}"
        ) from error

    if not page_count or not any(page.text.strip() for page in pages):
        raise PaperUnusableError(f"No extractable text in {paper.arxiv_id}.")

    return ParsedPaper(
        arxiv_id=paper.arxiv_id,
        pages=pages,
        page_count=page_count,
    )


def fetch_parsed_paper(paper: PaperMetadata) -> ParsedPaper:
    """Download and parse one paper."""

    return parse_paper(paper, download_paper(paper))
