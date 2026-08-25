"""Score the four retrieval strategies against the frozen labels.

Runs at the depth the pools were judged at. Pool depth and evaluation depth are both
`POOL_DEPTH`, which is what makes the numbers trustworthy: every chunk that can reach
a ranked list here was judged, so nothing is scored as irrelevant merely because
nobody looked at it. `check_coverage` enforces that rather than trusting it.

Results are grouped by question kind because the two kinds behave differently. Facet
questions have diffuse relevance — several chunks genuinely answer "what are the main
findings?" — so recall@5 is capped well below 1.0 for them and is reported with its
ceiling. The paper-specific questions have small relevant sets and no measured pooling
bias, so their recall is the figure to read.

`--multi-query` is deliberately absent. It expands each question into model-generated
paraphrases, which retrieve chunks the pools never contained, so `check_coverage`
would fire and the recall it produced would be understated for reasons unrelated to
the technique.
"""

import argparse
import json

from arxiv_reviewer.rag import RETRIEVER_KINDS, get_retriever

from .build_index import ensure_index, pool_chunk_ids
from .config import (
    EVAL_THREAD_ID,
    INDEX_DIR,
    POOL_DEPTH,
    POOLS_FILE,
    QUESTIONS_FILE,
    RESULTS_DIR,
    RETRIEVAL_LABELS,
)
from .metrics import (
    mean,
    ndcg_at,
    paired_bootstrap,
    recall_at,
    recall_ceiling,
    reciprocal_rank,
)

RESULTS_FILE = RESULTS_DIR / "retrieval.json"

# Must match `evals.build.pools`. Different values score a different ranking than the
# one the pool was built from, which makes the comparison against the labels invalid.
EVAL_K = POOL_DEPTH
FETCH_K = max(POOL_DEPTH * 2, 20)
RECALL_K = 5

COMPARISONS = (
    ("dense", "bm25"),
    ("dense", "hybrid"),
    ("dense", "hybrid-rerank"),
    ("hybrid", "hybrid-rerank"),
)

NDCG_KEY = f"ndcg@{EVAL_K}"
RECALL_KEY = f"recall@{RECALL_K}"
CEILING_KEY = f"recall@{RECALL_K}_ceiling"
METRIC_KEYS = ("mrr", NDCG_KEY, RECALL_KEY, CEILING_KEY)


def check_coverage(question_id: str, ranked: list[str], judged: set[str]) -> None:
    """Stop the run if a retrieved chunk was never judged."""

    unjudged = [chunk_id for chunk_id in ranked if chunk_id not in judged]
    if not unjudged:
        return

    raise SystemExit(
        f"{question_id}: retrieved unjudged chunks {unjudged}.\n"
        "The labels no longer cover what the retrievers return, so recall would be "
        "understated. Rebuild the pools and label the new candidates. Do not raise "
        "the cutoff to silence this."
    )


def score_question(ranked: list[str], relevant: dict[str, int]) -> dict:
    """Compute every metric for one question's ranking."""

    return {
        "mrr": reciprocal_rank(ranked, relevant),
        NDCG_KEY: ndcg_at(ranked, relevant, EVAL_K),
        RECALL_KEY: recall_at(ranked, relevant, RECALL_K),
        CEILING_KEY: recall_ceiling(relevant, RECALL_K),
        "relevant_chunks": len(relevant),
    }


def summarize(scores: list[dict]) -> dict:
    """Average per-question scores into one summary."""

    summary = {key: mean([score[key] for score in scores]) for key in METRIC_KEYS}
    summary["questions"] = len(scores)
    return summary


def group_by(scores: dict[str, dict], field: str) -> dict[str, dict]:
    """Summarize scores grouped by one of their fields, skipping empty groups."""

    groups: dict[str, list[dict]] = {}
    for score in scores.values():
        value = score.get(field)
        if value is not None:
            groups.setdefault(value, []).append(score)

    return {name: summarize(group) for name, group in sorted(groups.items())}


def run_retriever(
    kind: str,
    questions: list[dict],
    labels: dict,
    judged: dict[str, set[str]],
) -> dict:
    """Retrieve for every question with one strategy and score the rankings."""

    per_question: dict[str, dict] = {}

    for question in questions:
        question_id = question["question_id"]
        retriever = get_retriever(
            EVAL_THREAD_ID,
            kind=kind,
            k=EVAL_K,
            fetch_k=FETCH_K,
            arxiv_id=question["arxiv_id"],
            data_dir=INDEX_DIR,
        )
        ranked = [
            document.metadata["chunk_id"]
            for document in retriever.invoke(question["text"])
        ]
        check_coverage(question_id, ranked, judged[question_id])

        per_question[question_id] = {
            "kind": question["kind"],
            "facet": question["facet"],
            "retrieved": ranked,
            **score_question(ranked, labels[question_id]),
        }

    return {
        "overall": summarize(list(per_question.values())),
        "by_kind": group_by(per_question, "kind"),
        "by_facet": group_by(per_question, "facet"),
        "per_question": per_question,
    }


def compare(results: dict) -> list[dict]:
    """Bootstrap the per-question differences between retriever pairs.

    Without this the ablation table invites overclaiming: at fifty questions a gap of
    two or three points of nDCG is indistinguishable from noise, and reporting it as
    an improvement would be exactly the suspiciously clean win the dataset was built
    to avoid.
    """

    rows = []
    for baseline, variant in COMPARISONS:
        if baseline not in results or variant not in results:
            continue

        base_scores = results[baseline]["per_question"]
        variant_scores = results[variant]["per_question"]

        for metric, kind in (
            ("mrr", None),
            (NDCG_KEY, None),
            (RECALL_KEY, "specific"),
            (RECALL_KEY, "facet"),
        ):
            question_ids = [
                question_id
                for question_id, score in base_scores.items()
                if kind is None or score["kind"] == kind
            ]
            differences = [
                variant_scores[question_id][metric] - base_scores[question_id][metric]
                for question_id in question_ids
            ]
            difference, low, high = paired_bootstrap(differences)

            rows.append(
                {
                    "baseline": baseline,
                    "variant": variant,
                    "metric": metric,
                    "questions": kind or "all",
                    "n": len(differences),
                    "difference": difference,
                    "ci_low": low,
                    "ci_high": high,
                    "distinguishable": not (low <= 0.0 <= high),
                    "wins": sum(1 for value in differences if value > 1e-9),
                    "losses": sum(1 for value in differences if value < -1e-9),
                }
            )

    return rows


def print_comparisons(rows: list[dict]) -> None:
    """Print which differences survive resampling and which do not."""

    header = f"{'comparison':<34}{'metric':<18}{'diff':>8}{'95% CI':>20}{'w/l':>9}"
    print(f"\n{header}\n{'-' * len(header)}")

    for row in rows:
        pair = f"{row['baseline']} -> {row['variant']}"
        metric = f"{row['metric']}"
        if row["questions"] != "all":
            metric += f" ({row['questions'][:4]})"
        marker = "  *" if row["distinguishable"] else ""
        print(
            f"{pair:<34}{metric:<18}{row['difference']:>+8.3f}"
            f"   [{row['ci_low']:>+6.3f}, {row['ci_high']:>+6.3f}]"
            f"{row['wins']:>5}/{row['losses']}{marker}"
        )

    print(
        "\n* marks a 95% interval that excludes zero. Unmarked rows are differences\n"
        "this benchmark cannot distinguish from noise at 50 questions."
    )


def print_ablation(results: dict) -> None:
    """Print the four-way ablation table."""

    header = (
        f"{'retriever':<16}{'MRR':>8}{NDCG_KEY.upper():>10}"
        f"{'R@5 spec':>11}{'R@5 facet':>12}"
    )
    print(f"\n{header}\n{'-' * len(header)}")

    for kind in RETRIEVER_KINDS:
        if kind not in results:
            continue
        entry = results[kind]
        specific = entry["by_kind"].get("specific", {})
        facet = entry["by_kind"].get("facet", {})
        print(
            f"{kind:<16}{entry['overall']['mrr']:>8.3f}{entry['overall'][NDCG_KEY]:>10.3f}"
            f"{specific.get(RECALL_KEY, 0.0):>11.3f}{facet.get(RECALL_KEY, 0.0):>12.3f}"
        )

    sample = next(iter(results.values()))
    print("-" * len(header))
    print(
        f"{'(ceiling)':<16}{'':>8}{'':>10}"
        f"{sample['by_kind']['specific'][CEILING_KEY]:>11.3f}"
        f"{sample['by_kind']['facet'][CEILING_KEY]:>12.3f}"
    )
    print(
        "\nRecall is split by question kind on purpose: facet questions have more\n"
        "relevant chunks than the cutoff has slots, so their ceiling is not 1.0."
    )


def print_by_facet(results: dict) -> None:
    """Print nDCG per facet, showing which questions are hardest to retrieve for."""

    facets = sorted({facet for entry in results.values() for facet in entry["by_facet"]})
    header = f"\n{NDCG_KEY} by facet" + "".join(f"{kind[:11]:>12}" for kind in results)
    print(header)
    print("-" * (22 + 12 * len(results)))

    for facet in facets:
        row = "".join(
            f"{entry['by_facet'].get(facet, {}).get(NDCG_KEY, 0.0):>12.3f}"
            for entry in results.values()
        )
        print(f"{facet:<22}{row}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retriever",
        choices=RETRIEVER_KINDS,
        action="append",
        help="score only this strategy (repeatable); default is all four",
    )
    args = parser.parse_args()

    added, total = ensure_index()
    print(f"index: {total} chunks ({added} newly embedded)")

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    labels = json.loads(RETRIEVAL_LABELS.read_text(encoding="utf-8"))
    judged = pool_chunk_ids()
    kinds = args.retriever or list(RETRIEVER_KINDS)

    results = {}
    for kind in kinds:
        print(f"  scoring {kind:<14} over {len(questions)} questions ...")
        results[kind] = run_retriever(kind, questions, labels, judged)

    comparisons = compare(results)

    payload = {
        "config": {
            "eval_k": EVAL_K,
            "fetch_k": FETCH_K,
            "recall_k": RECALL_K,
            "pool_depth": POOL_DEPTH,
            "relevance": "grade >= 1 counts as relevant; nDCG uses the graded 0/1/2",
            "coverage": "every retrieved chunk verified present in the judged pool",
            "excluded": "multi-query: its paraphrases retrieve chunks outside the pools",
        },
        "dataset": {
            "questions": len(questions),
            "pools": POOLS_FILE.name,
            "labels": RETRIEVAL_LABELS.name,
        },
        "retrievers": results,
        "comparisons": comparisons,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print_ablation(results)
    print_by_facet(results)
    print_comparisons(comparisons)
    print(f"\nwrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
