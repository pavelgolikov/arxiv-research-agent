"""Generate the offline labeling page with the frozen datasets embedded.

The data is inlined rather than fetched so the page works from `file://`, where
browsers refuse cross-origin reads of local JSON.
"""

import json

from .config import (
    CANDIDATES_FILE,
    EVALS_DIR,
    POOLS_FILE,
    RETRIEVAL_LABELS,
    SCREENING_LABELS,
)

TEMPLATE = EVALS_DIR / "label_tool_template.html"
OUTPUT = EVALS_DIR / "label_tool.html"


def build_seed() -> dict:
    """Load committed labels so the page can restore work the browser has lost."""

    seed: dict = {"retrieval": {}, "screening": {}, "reviewed": {}}

    for mode, path in (("retrieval", RETRIEVAL_LABELS), ("screening", SCREENING_LABELS)):
        marks = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        seed[mode] = marks
        # A question present in a committed label file was reviewed to completion.
        seed["reviewed"][mode] = {question_id: True for question_id in marks}

    return seed


def main() -> None:
    pools = json.loads(POOLS_FILE.read_text(encoding="utf-8"))
    frozen = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))

    screening = [
        {"query_id": query_id, "query": entry["query"], "candidates": entry["candidates"]}
        for query_id, entry in frozen.items()
    ]
    payload = {"retrieval": pools, "screening": screening}
    seed = build_seed()

    html = (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
        .replace("__SEED__", json.dumps(seed, ensure_ascii=False))
    )
    OUTPUT.write_text(html, encoding="utf-8")

    judgements = sum(len(pool["candidates"]) for pool in pools)
    judgements += sum(len(entry["candidates"]) for entry in screening)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    print(f"{len(pools)} retrieval questions, {len(screening)} screening queries")
    print(f"seeded {len(seed['reviewed']['retrieval'])} retrieval and "
          f"{len(seed['reviewed']['screening'])} screening questions as already done")
    print(f"{judgements} candidates available to judge")


if __name__ == "__main__":
    main()
