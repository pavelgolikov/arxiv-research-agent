"""Validate exported labels and split them into the committed label files.

Run this on the JSON exported from `label_tool.html`. It rejects labels that do
not correspond to the frozen data, reports coverage, and estimates how much
relevant material the candidate pools missed.
"""

import argparse
import json
from pathlib import Path

from .config import (
    CANDIDATES_FILE,
    LABELS_DIR,
    POOLS_FILE,
    RETRIEVAL_LABELS,
    SCREENING_LABELS,
)

VALID_LABELS = {0, 1, 2}


def check_retrieval(exported: dict, pools: list[dict]) -> tuple[dict, list[str]]:
    """Validate retrieval labels against the frozen pools."""

    problems: list[str] = []
    reviewed = exported.get("reviewed", {}).get("retrieval", {})
    labels = exported.get("retrieval", {})
    clean: dict[str, dict[str, int]] = {}

    for pool in pools:
        question_id = pool["question_id"]
        known = {candidate["chunk_id"] for candidate in pool["candidates"]}
        given = labels.get(question_id, {})

        if not reviewed.get(question_id):
            problems.append(f"{question_id}: not marked reviewed")

        kept: dict[str, int] = {}
        for chunk_id, value in given.items():
            if chunk_id not in known:
                problems.append(f"{question_id}: {chunk_id} is not in the pool")
                continue
            if value not in VALID_LABELS:
                problems.append(f"{question_id}: {chunk_id} has label {value!r}")
                continue
            if value:
                kept[chunk_id] = value

        if reviewed.get(question_id) and not kept:
            problems.append(f"{question_id}: reviewed but no relevant chunk marked")

        clean[question_id] = dict(sorted(kept.items()))

    return clean, problems


def check_screening(exported: dict, frozen: dict) -> tuple[dict, list[str]]:
    """Validate screening labels against the frozen candidate metadata."""

    problems: list[str] = []
    reviewed = exported.get("reviewed", {}).get("screening", {})
    labels = exported.get("screening", {})
    clean: dict[str, dict[str, int]] = {}

    for query_id, entry in frozen.items():
        known = {candidate["arxiv_id"] for candidate in entry["candidates"]}
        given = labels.get(query_id, {})

        if not reviewed.get(query_id):
            problems.append(f"{query_id}: not marked reviewed")

        kept: dict[str, int] = {}
        for arxiv_id in sorted(known):
            value = given.get(arxiv_id, 0)
            if value not in VALID_LABELS:
                problems.append(f"{query_id}: {arxiv_id} has label {value!r}")
                continue
            kept[arxiv_id] = value

        for arxiv_id in given:
            if arxiv_id not in known:
                problems.append(f"{query_id}: {arxiv_id} is not a frozen candidate")

        clean[query_id] = kept

    return clean, problems


def report_retrieval(clean: dict, pools: list[dict]) -> None:
    """Summarize retrieval label coverage and pooling bias."""

    by_id = {pool["question_id"]: pool for pool in pools}
    graded = [value for marks in clean.values() for value in marks.values()]
    relevant_counts = [len(marks) for marks in clean.values()]

    sampled_total = 0
    sampled_relevant = 0
    for question_id, marks in clean.items():
        for candidate in by_id[question_id]["candidates"]:
            if not candidate["sampled"]:
                continue
            sampled_total += 1
            if marks.get(candidate["chunk_id"], 0):
                sampled_relevant += 1

    print("\nRetrieval labels")
    print(f"  questions              : {len(clean)}")
    print(f"  relevant chunks marked : {len(graded)}")
    print(f"    grade 2 (answers)    : {graded.count(2)}")
    print(f"    grade 1 (partial)    : {graded.count(1)}")
    if relevant_counts:
        print(f"  per question           : min {min(relevant_counts)}, "
              f"median {sorted(relevant_counts)[len(relevant_counts)//2]}, "
              f"max {max(relevant_counts)}")
    empty = [q for q, marks in clean.items() if not marks]
    if empty:
        print(f"  questions with none    : {len(empty)} -> {empty[:5]}")

    print("\n  Pooling bias check")
    print(f"    randomly sampled unpooled chunks judged : {sampled_total}")
    print(f"    of those found relevant                 : {sampled_relevant}")
    if sampled_total:
        rate = sampled_relevant / sampled_total
        print(f"    estimated miss rate                     : {rate:.1%}")
        if rate > 0.05:
            print("    NOTE: pools look too shallow; recall will be overstated.")
        else:
            print("    Pools look deep enough to treat unjudged chunks as irrelevant.")


def report_screening(clean: dict) -> None:
    """Summarize screening label distribution."""

    print("\nScreening labels")
    for query_id, marks in clean.items():
        values = list(marks.values())
        print(f"  {query_id:<22} central {values.count(2):>2}  "
              f"related {values.count(1):>2}  irrelevant {values.count(0):>2}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="labels.json exported from the tool")
    parser.add_argument("--write", action="store_true", help="write the label files")
    args = parser.parse_args()

    exported = json.loads(args.export.read_text(encoding="utf-8"))
    pools = json.loads(POOLS_FILE.read_text(encoding="utf-8"))
    frozen = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))

    retrieval, retrieval_problems = check_retrieval(exported, pools)
    screening, screening_problems = check_screening(exported, frozen)

    report_retrieval(retrieval, pools)
    report_screening(screening)

    problems = retrieval_problems + screening_problems
    print(f"\nProblems: {len(problems)}")
    for problem in problems[:25]:
        print(f"  - {problem}")
    if len(problems) > 25:
        print(f"  ... and {len(problems) - 25} more")

    if args.write:
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        RETRIEVAL_LABELS.write_text(
            json.dumps(retrieval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        SCREENING_LABELS.write_text(
            json.dumps(screening, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {RETRIEVAL_LABELS}")
        print(f"wrote {SCREENING_LABELS}")
    else:
        print("\n(dry run — pass --write to save the label files)")


if __name__ == "__main__":
    main()
