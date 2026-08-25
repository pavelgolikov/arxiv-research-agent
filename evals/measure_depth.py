"""Does searching arXiv deeper recover more on-topic papers?

A live run at `--max-results 10` selected one paper of ten. `search_node` divides the
result budget across the planned queries, so three queries at ten results gave each
query only four slots: one weak query burns a third of the candidate pool regardless
of how good the others are.

This replays the exact queries that run planned, at a deeper per-query budget, and
screens everything with the real screener. Holding the queries fixed is the point —
re-planning them would confound search depth with query variation, and the planner is
a model call whose output moves between runs.

Papers are reported at each depth, so the marginal yield of going deeper is visible
rather than inferred.
"""

import argparse
import json
from pathlib import Path

from arxiv_reviewer.analysis import RELEVANCE_THRESHOLD, screen_candidate
from arxiv_reviewer.failures import describe, with_retries
from arxiv_reviewer.retrieval import search_arxiv
from arxiv_reviewer.review_types import ScreenTask
from arxiv_reviewer.workflow import persistent_graph, thread_config
from arxiv_reviewer.rag import DEFAULT_DATA_DIR

from .config import RESULTS_DIR

RESULTS_FILE = RESULTS_DIR / "search_depth.json"

SHALLOW = 4   # what max-results 10 across three queries actually allowed
DEEP = 10     # what max-results 30 would allow


def planned_queries(thread_id: str, data_dir: Path) -> tuple[str, list[str]]:
    """Recover the user query and the arXiv queries a completed run planned."""

    snapshot = persistent_graph(data_dir).get_state(thread_config(thread_id))
    if snapshot.created_at is None:
        raise SystemExit(f"Unknown thread: {thread_id}")

    values = snapshot.values
    return values.get("user_query", ""), list(values.get("search_queries", []))


def gather(queries: list[str], depth: int) -> list[dict]:
    """Fetch `depth` results per query, deduplicated, keeping the rank each came from."""

    seen: set[str] = set()
    papers: list[dict] = []

    for position, query in enumerate(queries):
        for rank, paper in enumerate(search_arxiv(query, depth)):
            if paper.arxiv_id in seen:
                continue
            seen.add(paper.arxiv_id)
            papers.append({"paper": paper, "query_index": position, "rank": rank})
        print(f"  query {position + 1}: {len(papers)} unique so far")

    return papers


def screen_all(papers: list[dict], user_query: str) -> list[dict]:
    """Score every gathered paper with the real screener."""

    scored = []
    for index, item in enumerate(papers):
        task = ScreenTask(
            paper=item["paper"], search_position=index, user_query=user_query
        )
        try:
            outcome = with_retries(lambda: screen_candidate(task))
            score, status, error = outcome.score, "ok", None
        except Exception as failure:
            score, status, error = 0, "failed", describe(failure)

        scored.append(
            {
                "arxiv_id": item["paper"].arxiv_id,
                "title": item["paper"].title,
                "query_index": item["query_index"],
                "rank": item["rank"],
                "score": score,
                "status": status,
                "error": error,
            }
        )
    return scored


def at_depth(scored: list[dict], depth: int) -> dict:
    """Restrict to the first `depth` results of each query and count what qualifies."""

    subset = [row for row in scored if row["rank"] < depth]
    relevant = [row for row in subset if row["status"] == "ok" and row["score"] >= RELEVANCE_THRESHOLD]

    return {
        "depth_per_query": depth,
        "candidates": len(subset),
        "relevant": len(relevant),
        "relevant_ids": [row["arxiv_id"] for row in relevant],
        "density": len(relevant) / len(subset) if subset else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", default="groundedness-fixed-1")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    user_query, queries = planned_queries(args.thread_id, args.data_dir)
    print(f"user query : {user_query}")
    for index, query in enumerate(queries):
        print(f"  query {index + 1}: {query}")

    print(f"\nfetching {DEEP} results per query")
    papers = gather(queries, DEEP)

    print(f"\nscreening {len(papers)} unique papers")
    scored = screen_all(papers, user_query)

    shallow, deep = at_depth(scored, SHALLOW), at_depth(scored, DEEP)
    payload = {
        "question": "does a deeper per-query budget recover more on-topic papers",
        "thread_id": args.thread_id,
        "user_query": user_query,
        "queries": queries,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "shallow": shallow,
        "deep": deep,
        "scored": scored,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\n{'per-query depth':<18}{'candidates':>12}{'relevant':>10}{'density':>10}")
    print("-" * 50)
    for label, stats in (("shallow", shallow), ("deep", deep)):
        print(f"{stats['depth_per_query']:<18}{stats['candidates']:>12}"
              f"{stats['relevant']:>10}{stats['density']:>10.0%}")

    gained = set(deep["relevant_ids"]) - set(shallow["relevant_ids"])
    print(f"\nrecovered by going deeper: {len(gained)} paper(s)")
    for row in scored:
        if row["arxiv_id"] in gained:
            print(f"  q{row['query_index'] + 1} rank {row['rank']:>2}  "
                  f"score {row['score']}  {row['title'][:58]}")

    print(f"\nscore distribution by rank band")
    for low, high in ((0, 4), (4, 10)):
        band = [r for r in scored if low <= r["rank"] < high]
        good = sum(1 for r in band if r["score"] >= RELEVANCE_THRESHOLD)
        print(f"  ranks {low}-{high - 1}: {good}/{len(band)} scored >= {RELEVANCE_THRESHOLD}")

    print(f"\nwrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
