"""Download, parse, and chunk the frozen retrieval corpus.

The chunk file is the durable artifact: labels reference `chunk_id`, and those
identifiers are derived from position, so rebuilding reproduces them exactly.
"""

import json

from arxiv_reviewer.rag import chunk_pages, index_paper
from arxiv_reviewer.retrieval import fetch_parsed_paper
from arxiv_reviewer.review_types import PaperMetadata

from .config import (
    CANDIDATES_FILE,
    CHUNKS_FILE,
    CORPUS_PAPERS,
    DATA_DIR,
    EVAL_THREAD_ID,
    INDEX_DIR,
    PAPERS_FILE,
)


def load_selected() -> list[tuple[str, dict]]:
    """Return the frozen candidate record for each chosen corpus paper."""

    frozen = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))
    selected = []

    for query_id, arxiv_id in CORPUS_PAPERS.items():
        matches = [
            candidate
            for candidate in frozen[query_id]["candidates"]
            if candidate["arxiv_id"] == arxiv_id
        ]
        if not matches:
            raise SystemExit(
                f"{arxiv_id} is not among the frozen candidates for {query_id}."
            )
        selected.append((query_id, matches[0]))

    return selected


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    papers = []
    records = []

    for query_id, candidate in load_selected():
        metadata = PaperMetadata(
            arxiv_id=candidate["arxiv_id"],
            title=candidate["title"],
            authors=candidate["authors"],
            abstract=candidate["abstract"],
            published=candidate["published"],
            pdf_url=candidate["pdf_url"],
            entry_url=candidate["entry_url"],
        )
        parsed = fetch_parsed_paper(metadata)
        documents = chunk_pages(metadata.arxiv_id, parsed.pages)

        index_paper(EVAL_THREAD_ID, documents, data_dir=INDEX_DIR)

        papers.append(
            {
                "query_id": query_id,
                "arxiv_id": metadata.arxiv_id,
                "title": metadata.title,
                "authors": metadata.authors,
                "published": metadata.published,
                "entry_url": metadata.entry_url,
                "page_count": parsed.page_count,
                "chunk_count": len(documents),
            }
        )
        records.extend(
            {
                "chunk_id": document.metadata["chunk_id"],
                "arxiv_id": document.metadata["arxiv_id"],
                "page_number": document.metadata["page_number"],
                "text": document.page_content,
            }
            for document in documents
        )
        print(
            f"  {metadata.arxiv_id:<14} {parsed.page_count:>3} pages "
            f"{len(documents):>4} chunks  {metadata.title[:52]}"
        )

    PAPERS_FILE.write_text(
        json.dumps(papers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with CHUNKS_FILE.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nwrote {PAPERS_FILE}")
    print(f"wrote {CHUNKS_FILE} ({len(records)} chunks)")
    print(f"indexed into {INDEX_DIR}")


if __name__ == "__main__":
    main()
