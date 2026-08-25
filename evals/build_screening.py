"""Freeze arXiv candidate metadata so screening labels stay reproducible.

The literal query text is searched directly, bypassing the LLM query planner, so
rebuilding this file does not depend on a model's output.
"""

import json
import time

from arxiv_reviewer.retrieval import search_arxiv

from .config import CANDIDATES_FILE, CANDIDATES_PER_QUERY, DATA_DIR, EVAL_QUERIES


def build() -> dict:
    """Search each evaluation query and record its candidate metadata."""

    frozen = {}

    for index, query in enumerate(EVAL_QUERIES):
        if index > 0:
            time.sleep(3.0)

        papers = search_arxiv(query["text"], CANDIDATES_PER_QUERY)
        frozen[query["query_id"]] = {
            "query": query["text"],
            "candidates": [
                {
                    "search_position": position,
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors[:6],
                    "published": paper.published,
                    "abstract": paper.abstract,
                    "entry_url": paper.entry_url,
                    "pdf_url": paper.pdf_url,
                }
                for position, paper in enumerate(papers)
            ],
        }
        print(f"  {query['query_id']:<22} {len(papers)} candidates")

    return frozen


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frozen = build()
    CANDIDATES_FILE.write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    total = sum(len(entry["candidates"]) for entry in frozen.values())
    print(f"\nwrote {CANDIDATES_FILE} ({total} candidates)")


if __name__ == "__main__":
    main()
