"""Measure abstract-only screening against the frozen labels, and pick a threshold.

Screening decides which papers ever get downloaded, so its errors are the most
expensive kind in the pipeline: a central paper rejected here can never be recovered
later. `RELEVANCE_THRESHOLD` was chosen by hand; this runner sweeps it through the
real `select_papers_node` and reports what each value would have selected.

Scoring the candidates costs one model call each, so the scores are cached. Metrics
can then be recomputed as often as wanted without paying again, and always against
the same judgments.

Two of the seven queries came back twelve-central-out-of-twelve, which makes their
precision saturate at 1.0 no matter what is selected. Results are therefore reported
per query as well as pooled, so a saturated query cannot quietly flatter the average.
"""

import argparse
import json

from arxiv_reviewer.analysis import RELEVANCE_THRESHOLD, screen_candidate, select_papers_node
from arxiv_reviewer.failures import describe, with_retries
from arxiv_reviewer.gemini_client import GEMINI_MODEL
from arxiv_reviewer.review_types import PaperMetadata, ScreenOutcome, ScreenTask

from .config import CANDIDATES_FILE, RESULTS_DIR, SCREENING_LABELS
from .metrics import mean

SCORES_FILE = RESULTS_DIR / "screening_scores.json"
RESULTS_FILE = RESULTS_DIR / "screening.json"

# The CLI default. Screening is measured at the size a real review actually asks for.
TARGET_PAPERS = 4
THRESHOLDS = (1, 2, 3, 4, 5)

CENTRAL = 2
RELATED = 1


def score_candidates(frozen: dict) -> dict:
    """Run the real screener over every frozen candidate."""

    scores: dict[str, dict] = {}

    for query_id, entry in frozen.items():
        scores[query_id] = {}
        for candidate in entry["candidates"]:
            task = ScreenTask(
                paper=PaperMetadata(
                    arxiv_id=candidate["arxiv_id"],
                    title=candidate["title"],
                    authors=candidate["authors"],
                    abstract=candidate["abstract"],
                    published=candidate["published"],
                    pdf_url=candidate["pdf_url"],
                    entry_url=candidate["entry_url"],
                ),
                search_position=candidate["search_position"],
                user_query=entry["query"],
            )
            try:
                outcome = with_retries(lambda: screen_candidate(task))
                record = {
                    "search_position": task["search_position"],
                    "score": outcome.score,
                    "reason": outcome.reason,
                    "status": "ok",
                }
            except Exception as error:
                record = {
                    "search_position": task["search_position"],
                    "score": 0,
                    "reason": "",
                    "status": "failed",
                    "error": describe(error),
                }

            scores[query_id][candidate["arxiv_id"]] = record

        ok = sum(1 for r in scores[query_id].values() if r["status"] == "ok")
        print(f"  {query_id:<24} scored {ok}/{len(entry['candidates'])}")

    return scores


def load_scores(frozen: dict, refresh: bool) -> dict:
    """Return cached scores, or produce and cache them."""

    if not refresh and SCORES_FILE.exists():
        cached = json.loads(SCORES_FILE.read_text(encoding="utf-8"))
        if set(cached.get("scores", {})) == set(frozen):
            print(f"scores: reusing {SCORES_FILE.name} (no model calls)")
            return cached["scores"]
        print("scores: cache does not cover every query; rescoring")

    print(f"scores: calling the screener ({sum(len(e['candidates']) for e in frozen.values())} calls)")
    scores = score_candidates(frozen)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_FILE.write_text(
        json.dumps({"model": GEMINI_MODEL, "scores": scores}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {SCORES_FILE}")
    return scores


def evaluate_query(
    query_scores: dict,
    query_labels: dict[str, int],
    threshold: int,
) -> dict:
    """Select papers at one threshold and score the selection against the labels."""

    evaluations = [
        ScreenOutcome(
            arxiv_id=arxiv_id,
            search_position=record["search_position"],
            score=record["score"],
            status=record["status"],
        )
        for arxiv_id, record in query_scores.items()
    ]
    # The real selection rule, not a copy of it: filter by threshold, sort by score
    # then original search position, truncate to the target.
    selected = select_papers_node(
        {
            "candidate_evaluations": evaluations,
            "target_papers": TARGET_PAPERS,
            "relevance_threshold": threshold,
        }
    )["selected_ids"]

    central = {a for a, label in query_labels.items() if label == CENTRAL}
    chosen_labels = [query_labels.get(arxiv_id, 0) for arxiv_id in selected]

    return {
        "threshold": threshold,
        "selected": selected,
        "selected_count": len(selected),
        "underfilled": len(selected) < TARGET_PAPERS,
        "precision_central": mean([float(v == CENTRAL) for v in chosen_labels]),
        "precision_related": mean([float(v >= RELATED) for v in chosen_labels]),
        "central_recall": len(central & set(selected)) / len(central) if central else 0.0,
        "central_recall_ceiling": (
            min(TARGET_PAPERS, len(central)) / len(central) if central else 0.0
        ),
        "central_available": len(central),
    }


def agreement(scores: dict, labels: dict) -> dict:
    """Report the mean model score for each human label class.

    If the 1-5 rubric is doing real work, these three means separate. If they do not,
    the threshold sweep is choosing between rankings that barely differ.
    """

    buckets: dict[str, list[float]] = {"irrelevant": [], "related": [], "central": []}
    names = {0: "irrelevant", RELATED: "related", CENTRAL: "central"}

    for query_id, query_scores in scores.items():
        for arxiv_id, record in query_scores.items():
            if record["status"] != "ok":
                continue
            label = labels[query_id].get(arxiv_id)
            if label in names:
                buckets[names[label]].append(float(record["score"]))

    return {
        name: {"mean_model_score": mean(values), "n": len(values)}
        for name, values in buckets.items()
    }


def confusion(scores: dict, labels: dict) -> dict:
    """Cross-tabulate the model's 1-5 score against the human 0/1/2 label."""

    names = {0: "irrelevant", RELATED: "related", CENTRAL: "central"}
    grid = {
        str(score): {name: 0 for name in names.values()} for score in range(1, 6)
    }

    for query_id, query_scores in scores.items():
        for arxiv_id, record in query_scores.items():
            if record["status"] != "ok":
                continue
            label = labels[query_id].get(arxiv_id)
            if label in names:
                grid[str(record["score"])][names[label]] += 1

    return grid


def recommend(sweep: dict) -> int:
    """Pick the threshold with the best central precision, then the best recall.

    Precision leads because a wrong paper wastes a download and six model calls and
    then occupies a slot in the review; under-filling only makes the review shorter.
    """

    def key(threshold: int) -> tuple[float, float, int, int]:
        pooled = sweep[str(threshold)]["pooled"]
        return (
            pooled["precision_central"],
            pooled["central_recall"],
            -pooled["underfilled_queries"],
            # Ties go to the stricter threshold. Equal performance on seven queries is
            # not evidence that the looser value is equally safe on unseen ones.
            threshold,
        )

    return max(THRESHOLDS, key=key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="rescore every candidate (84 model calls)"
    )
    args = parser.parse_args()

    frozen = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))
    labels = json.loads(SCREENING_LABELS.read_text(encoding="utf-8"))
    scores = load_scores(frozen, refresh=args.refresh)

    sweep: dict[str, dict] = {}
    for threshold in THRESHOLDS:
        per_query = {
            query_id: evaluate_query(scores[query_id], labels[query_id], threshold)
            for query_id in frozen
        }
        sweep[str(threshold)] = {
            "per_query": per_query,
            "pooled": {
                "precision_central": mean(
                    [q["precision_central"] for q in per_query.values()]
                ),
                "precision_related": mean(
                    [q["precision_related"] for q in per_query.values()]
                ),
                "central_recall": mean([q["central_recall"] for q in per_query.values()]),
                "central_recall_ceiling": mean(
                    [q["central_recall_ceiling"] for q in per_query.values()]
                ),
                "underfilled_queries": sum(1 for q in per_query.values() if q["underfilled"]),
            },
        }

    recommended = recommend(sweep)
    payload = {
        "config": {
            "target_papers": TARGET_PAPERS,
            "thresholds": list(THRESHOLDS),
            "current_default": RELEVANCE_THRESHOLD,
            "labels": "0 irrelevant, 1 related, 2 central",
            "model": GEMINI_MODEL,
        },
        "recommended_threshold": recommended,
        "agreement": agreement(scores, labels),
        "confusion": confusion(scores, labels),
        "sweep": sweep,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print_report(payload, frozen, labels)
    print(f"\nwrote {RESULTS_FILE}")


def print_report(payload: dict, frozen: dict, labels: dict) -> None:
    """Print the sweep, the per-query detail, and the rubric agreement."""

    print(f"\n{'threshold':<11}{'P@4 central':>13}{'P@4 related':>13}"
          f"{'central recall':>16}{'ceiling':>9}{'underfilled':>13}")
    print("-" * 75)
    for threshold in THRESHOLDS:
        pooled = payload["sweep"][str(threshold)]["pooled"]
        mark = "  <- current" if threshold == payload["config"]["current_default"] else ""
        mark += "  RECOMMENDED" if threshold == payload["recommended_threshold"] else ""
        print(
            f"{threshold:<11}{pooled['precision_central']:>13.3f}"
            f"{pooled['precision_related']:>13.3f}{pooled['central_recall']:>16.3f}"
            f"{pooled['central_recall_ceiling']:>9.3f}"
            f"{pooled['underfilled_queries']:>13}{mark}"
        )

    best = payload["sweep"][str(payload["recommended_threshold"])]["per_query"]
    print(f"\nper query at threshold {payload['recommended_threshold']}")
    print(f"{'query':<24}{'central':>9}{'P@4':>8}{'recall':>9}{'ceiling':>9}{'picked':>8}")
    print("-" * 67)
    for query_id, result in best.items():
        saturated = " (saturated)" if result["central_available"] >= len(
            frozen[query_id]["candidates"]
        ) else ""
        print(
            f"{query_id:<24}{result['central_available']:>9}"
            f"{result['precision_central']:>8.2f}{result['central_recall']:>9.2f}"
            f"{result['central_recall_ceiling']:>9.2f}{result['selected_count']:>8}"
            f"{saturated}"
        )

    print("\nmodel score against human label")
    print(f"  {'':>7}{'irrelevant':>12}{'related':>9}{'central':>9}")
    for score, row in payload["confusion"].items():
        print(
            f"  score{score}{row['irrelevant']:>12}{row['related']:>9}{row['central']:>9}"
        )

    print("\nmean model score by human label")
    for name, stats in payload["agreement"].items():
        print(f"  {name:<12}{stats['mean_model_score']:>6.2f}  (n={stats['n']})")


if __name__ == "__main__":
    main()
