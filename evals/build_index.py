"""Rebuild the evaluation vector index from the committed chunk file.

`evals/index/` is not committed, so a fresh clone has no index and the retrieval
ablation has nothing to run against. Rebuilding from `corpus_chunks.jsonl` — which
is committed — keeps the indexed text identical to what was labelled, without
re-downloading or re-parsing a single PDF.

Only missing chunks are embedded. That matters: the labels describe the union of
what the four retrievers returned on the day the pools were built, so a chunk that
keeps its original vector keeps its original rank. Re-embedding a chunk that is
already indexed would risk moving it for no benefit, which is what `--force` is
for and why it is not the default.

`--verify` replays all four retrievers over all fifty questions and confirms that
every chunk they return was actually judged. That is the assumption the metrics
rest on: pool depth and evaluation depth are both ten, so no unjudged chunk can
enter a ranked list and be silently scored as irrelevant. If drift ever breaks
that, this fails loudly instead of quietly lowering recall.
"""

import argparse
import json

from langchain_chroma import Chroma
from langchain_core.documents import Document

from arxiv_reviewer.rag import (
    RETRIEVER_KINDS,
    chroma_client,
    collection_name,
    get_retriever,
    index_paper,
)

from .config import (
    CHUNKS_FILE,
    EVAL_THREAD_ID,
    INDEX_DIR,
    POOL_DEPTH,
    POOLS_FILE,
    RESULTS_DIR,
)

COVERAGE_FILE = RESULTS_DIR / "index_coverage.json"


def load_corpus() -> list[Document]:
    """Read the committed chunks back as the documents that were indexed.

    The metadata keys rebuilt here are the ones `chunk_pages` originally attached;
    retrieval filters on `arxiv_id` and every metric keys off `chunk_id`.
    """

    documents = []
    with CHUNKS_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            documents.append(
                Document(
                    page_content=record["text"],
                    metadata={
                        "arxiv_id": record["arxiv_id"],
                        "page_number": record["page_number"],
                        "chunk_id": record["chunk_id"],
                    },
                )
            )
    return documents


def indexed_ids() -> set[str]:
    """Return the chunk IDs already present in the index.

    Opened without an embedding function so this stays a local read: checking what
    is indexed needs no API key and makes no network call.
    """

    store = Chroma(
        collection_name=collection_name(EVAL_THREAD_ID),
        client=chroma_client(INDEX_DIR),
    )
    return set(store.get(include=[])["ids"])


def ensure_index(force: bool = False) -> tuple[int, int]:
    """Index any corpus chunk the store is missing; return (added, total)."""

    documents = load_corpus()

    if force:
        client = chroma_client(INDEX_DIR)
        try:
            client.delete_collection(collection_name(EVAL_THREAD_ID))
        except Exception:
            pass
        present: set[str] = set()
    else:
        present = indexed_ids()

    missing = [
        document
        for document in documents
        if document.metadata["chunk_id"] not in present
    ]
    if missing:
        index_paper(EVAL_THREAD_ID, missing, data_dir=INDEX_DIR)

    return len(missing), len(documents)


def pool_chunk_ids() -> dict[str, set[str]]:
    """Return the judged chunk IDs for each question."""

    pools = json.loads(POOLS_FILE.read_text(encoding="utf-8"))
    return {
        pool["question_id"]: {candidate["chunk_id"] for candidate in pool["candidates"]}
        for pool in pools
    }


def retrieved_ids(question: dict, kind: str) -> list[str]:
    """Retrieve for one question with one strategy, at pool depth.

    The retriever arguments must match `evals.build.pools` exactly. If they drift, this
    check compares a different ranking against the pool and means nothing.
    """

    retriever = get_retriever(
        EVAL_THREAD_ID,
        kind=kind,
        k=POOL_DEPTH,
        fetch_k=max(POOL_DEPTH * 2, 20),
        arxiv_id=question["arxiv_id"],
        data_dir=INDEX_DIR,
    )
    return [
        document.metadata["chunk_id"]
        for document in retriever.invoke(question["text"])
    ]


def verify() -> dict:
    """Confirm every chunk the retrievers return at pool depth was judged."""

    pools = json.loads(POOLS_FILE.read_text(encoding="utf-8"))
    judged = pool_chunk_ids()

    checked = 0
    unjudged: list[dict] = []

    for number, question in enumerate(pools, start=1):
        question_id = question["question_id"]
        misses = set()

        for kind in RETRIEVER_KINDS:
            for chunk_id in retrieved_ids(question, kind):
                checked += 1
                if chunk_id not in judged[question_id]:
                    misses.add(chunk_id)
                    unjudged.append(
                        {
                            "question_id": question_id,
                            "retriever": kind,
                            "chunk_id": chunk_id,
                        }
                    )

        flag = f"  {len(misses)} UNJUDGED" if misses else ""
        print(f"  [{number:>2}/{len(pools)}] {question_id:<36} ok{flag}")

    return {
        "pool_depth": POOL_DEPTH,
        "retrievers": list(RETRIEVER_KINDS),
        "questions": len(pools),
        "retrieved_chunks_checked": checked,
        "unjudged_retrieved": unjudged,
        "complete": not unjudged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="drop the collection and re-embed every chunk (may shift dense ranks)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="replay every retriever and confirm all retrieved chunks were judged",
    )
    args = parser.parse_args()

    added, total = ensure_index(force=args.force)
    print(f"corpus chunks : {total}")
    print(f"newly indexed : {added}")
    print(f"already present: {total - added}")
    print(f"index         : {INDEX_DIR / 'chroma'}")

    if not args.verify:
        print("\n(pass --verify to confirm the pools still cover every retrieved chunk)")
        return

    print(f"\nVerifying pool coverage at depth {POOL_DEPTH}")
    report = verify()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    COVERAGE_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nretrieved chunks checked : {report['retrieved_chunks_checked']}")
    print(f"unjudged among them      : {len(report['unjudged_retrieved'])}")
    print(f"wrote {COVERAGE_FILE}")

    if not report["complete"]:
        raise SystemExit(
            "Retrieved chunks are missing from the pools. The labels no longer "
            "cover what the retrievers return, so recall would be understated. "
            "Rebuild the pools and label the new candidates before scoring."
        )
    print("\nPools cover every retrieved chunk; metrics score only judged chunks.")


if __name__ == "__main__":
    main()
