"""Freeze arXiv candidate metadata so screening labels stay reproducible.

The literal query text is searched directly, bypassing the LLM query planner, so
rebuilding this file does not depend on a model's output.

Queries already present in the file are kept as they are and only new ones are
searched. arXiv rankings drift, so re-searching a frozen query would silently
swap candidates out from under labels that already reference them.
"""

import argparse
import json
import time

from arxiv_reviewer.retrieval import search_arxiv

from ..config import CANDIDATES_FILE, CANDIDATES_PER_QUERY, DATA_DIR, EVAL_QUERIES


def build(existing: dict | None = None) -> dict:
    """Search any query missing from `existing` and record its candidate metadata."""

    frozen = dict(existing or {})
    pending = [query for query in EVAL_QUERIES if query["query_id"] not in frozen]

    for query_id in frozen:
        print(f"  {query_id:<22} kept ({len(frozen[query_id]['candidates'])} candidates)")

    for index, query in enumerate(pending):
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
        print(f"  {query['query_id']:<22} {len(papers)} candidates (new)")

    order = [query["query_id"] for query in EVAL_QUERIES]
    return {query_id: frozen[query_id] for query_id in order if query_id in frozen}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-search every query, discarding frozen candidates (invalidates labels)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if not args.refresh and CANDIDATES_FILE.exists():
        existing = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))
    frozen = build(existing)
    CANDIDATES_FILE.write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    total = sum(len(entry["candidates"]) for entry in frozen.values())
    print(f"\nwrote {CANDIDATES_FILE} ({total} candidates)")


if __name__ == "__main__":
    main()
