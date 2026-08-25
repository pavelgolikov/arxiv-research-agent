"""Generate the offline labeling page with the frozen datasets embedded.

The data is inlined rather than fetched so the page works from `file://`, where
browsers refuse cross-origin reads of local JSON.
"""

import json

from .config import CANDIDATES_FILE, EVALS_DIR, POOLS_FILE

TEMPLATE = EVALS_DIR / "label_tool_template.html"
OUTPUT = EVALS_DIR / "label_tool.html"


def main() -> None:
    pools = json.loads(POOLS_FILE.read_text(encoding="utf-8"))
    frozen = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))

    screening = [
        {"query_id": query_id, "query": entry["query"], "candidates": entry["candidates"]}
        for query_id, entry in frozen.items()
    ]
    payload = {"retrieval": pools, "screening": screening}

    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(payload, ensure_ascii=False)
    )
    OUTPUT.write_text(html, encoding="utf-8")

    judgements = sum(len(pool["candidates"]) for pool in pools)
    judgements += sum(len(entry["candidates"]) for entry in screening)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    print(f"{len(pools)} retrieval questions, {len(screening)} screening queries")
    print(f"{judgements} candidates available to judge")


if __name__ == "__main__":
    main()
