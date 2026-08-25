"""Pool retrieval candidates so labelling stays tractable.

For each question every retriever contributes its top results, and the union is
what gets judged. Judging the union rather than the whole paper is what keeps the
work to roughly a dozen decisions per question instead of several hundred.

Pooling has a known bias: a chunk no retriever ever returns is never labelled, so
it silently counts as irrelevant and recall is overstated. To measure that bias
rather than ignore it, a few chunks that no retriever returned are sampled into
the pool as well. If those turn out to be relevant, the pool was too shallow.
"""

import json
import random

from arxiv_reviewer.rag import RETRIEVER_KINDS, get_retriever

from ..config import (
    CHUNKS_FILE,
    DATA_DIR,
    EVAL_THREAD_ID,
    INDEX_DIR,
    POOLS_FILE,
    POOL_DEPTH,
    QUESTIONS_FILE,
)

UNPOOLED_SAMPLE = 3
RANDOM_SEED = 20260808


def load_chunks() -> dict[str, dict]:
    """Load the frozen corpus keyed by chunk identifier."""

    chunks = {}
    with CHUNKS_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            chunks[record["chunk_id"]] = record
    return chunks


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    rng = random.Random(RANDOM_SEED)

    by_paper: dict[str, list[str]] = {}
    for chunk_id, record in chunks.items():
        by_paper.setdefault(record["arxiv_id"], []).append(chunk_id)

    pools = []
    for number, question in enumerate(questions, start=1):
        arxiv_id = question["arxiv_id"]
        found_by: dict[str, list[str]] = {}

        for kind in RETRIEVER_KINDS:
            retriever = get_retriever(
                EVAL_THREAD_ID,
                kind=kind,
                k=POOL_DEPTH,
                fetch_k=max(POOL_DEPTH * 2, 20),
                arxiv_id=arxiv_id,
                data_dir=INDEX_DIR,
            )
            for document in retriever.invoke(question["text"]):
                found_by.setdefault(document.metadata["chunk_id"], []).append(kind)

        unpooled = sorted(set(by_paper[arxiv_id]) - set(found_by))
        sampled = rng.sample(unpooled, min(UNPOOLED_SAMPLE, len(unpooled)))

        candidates = [
            {
                "chunk_id": chunk_id,
                "page_number": chunks[chunk_id]["page_number"],
                "text": chunks[chunk_id]["text"],
                "found_by": sorted(found_by.get(chunk_id, [])),
                "sampled": chunk_id in sampled,
            }
            for chunk_id in sorted(set(found_by) | set(sampled))
        ]
        # Present candidates in a fixed shuffled order. Ordering them by how many
        # retrievers agreed would leak the systems' opinions to the judge and bias
        # the very comparison this dataset exists to make.
        candidates.sort(key=lambda item: item["chunk_id"])
        random.Random(f"{RANDOM_SEED}:{question['question_id']}").shuffle(candidates)

        pools.append({**question, "candidates": candidates})
        print(
            f"  [{number:>2}/{len(questions)}] {question['question_id']:<34} "
            f"{len(found_by):>2} pooled + {len(sampled)} sampled"
        )

    POOLS_FILE.write_text(
        json.dumps(pools, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    total = sum(len(pool["candidates"]) for pool in pools)
    print(f"\nwrote {POOLS_FILE}")
    print(f"{len(pools)} questions, {total} candidates to judge "
          f"({total / len(pools):.1f} per question)")


if __name__ == "__main__":
    main()
