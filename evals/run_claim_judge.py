"""Score the automated support judge against the hand-labeled citations.

The pipeline drops any citation `judge_support` grades `0`. That check is a model
grading another model's output, which is worth exactly as much as the evidence that it
agrees with a reader. This runner is that evidence: it replays every hand-labeled
citation through the judge and reports where the two disagree.

Two numbers carry the claim, and they fail in opposite directions:

* **false-drop rate** — citations a human graded `1` or `2` that the judge grades `0`.
  This is what the check costs: correct work thrown out of the report.
* **catch rate** — citations a human graded `0` that the judge also grades `0`. This is
  what the check buys, and it can only be measured because the judge set contributes
  citations whose excerpt was swapped for another quote from the same paper. Scored on
  the uniform sample alone the rate would be undefined: none of those 40 is a `0`.

One difference from production is worth stating. Here the judge grades one citation per
call; in `analyze_facet` it grades a whole facet in one call and can read its items
against each other. Per-item is the leaner condition — there is no neighbouring verdict
to anchor to — but it is not identical, and a large batch effect would not show up here.

    python -m evals.run_claim_judge
"""

import argparse
import json
from pathlib import Path

from arxiv_reviewer.analysis import SUPPORT_THRESHOLD, judge_support
from arxiv_reviewer.failures import with_retries

from .config import LABELS_DIR, RESULTS_DIR
from .metrics import wilson_interval

LABELS_FILE = LABELS_DIR / "claim_support_labels.json"
RESULTS_FILE = RESULTS_DIR / "claim_judge.json"

GRADES = (2, 1, 0)


def grade_one(label: dict) -> int | None:
    """Put one labeled citation to the judge, retrying transient model failures."""

    grades = with_retries(lambda: judge_support([(label["claim"], label["excerpt"])]))
    return grades[0]


def kept(grade: int | None, threshold: int) -> bool:
    """Report whether a grade would keep the citation in the report.

    An unjudged citation is a drop, exactly as `apply_support_judge` treats it, so the
    measured rates describe the rule the pipeline really applies.
    """

    return grade is not None and grade >= threshold


def rate(numerator: int, denominator: int) -> dict:
    """Return a proportion with its Wilson interval, or nulls when nothing applies."""

    if not denominator:
        return {"n": 0, "of": 0, "rate": None, "ci": None}

    low, high = wilson_interval(numerator, denominator)
    return {
        "n": numerator,
        "of": denominator,
        "rate": numerator / denominator,
        "ci": [low, high],
    }


def confusion(scored: list[dict]) -> dict:
    """Count human grade against judge grade, with unjudged items in their own row."""

    grid = {
        str(human): {str(judge): 0 for judge in GRADES} | {"unjudged": 0}
        for human in GRADES
    }
    for entry in scored:
        judged = "unjudged" if entry["judge"] is None else str(entry["judge"])
        grid[str(entry["human"])][judged] += 1
    return grid


def score(scored: list[dict], threshold: int) -> dict:
    """Compute the two rates that decide whether the check is worth running."""

    supported = [entry for entry in scored if entry["human"] >= 1]
    unsupported = [entry for entry in scored if entry["human"] == 0]

    return {
        "threshold": threshold,
        "false_drop": rate(
            sum(1 for entry in supported if not kept(entry["judge"], threshold)),
            len(supported),
        ),
        "catch": rate(
            sum(1 for entry in unsupported if not kept(entry["judge"], threshold)),
            len(unsupported),
        ),
        "keep_agreement": rate(
            sum(
                1
                for entry in scored
                if kept(entry["judge"], threshold) == (entry["human"] >= threshold)
            ),
            len(scored),
        ),
    }


def summarize(scored: list[dict]) -> dict:
    """Build the payload: exact agreement, the shipped rule, and the stricter one."""

    exact = rate(
        sum(1 for entry in scored if entry["judge"] == entry["human"]), len(scored)
    )

    return {
        "config": {
            "labels": str(LABELS_FILE.relative_to(LABELS_DIR.parent.parent)),
            "threshold": SUPPORT_THRESHOLD,
            "batching": "one citation per call; production grades a facet per call",
            "note": (
                "Negatives are citations whose excerpt was swapped for another quote "
                "from the same paper, then hand-graded rather than assumed to be 0."
            ),
        },
        "scored": len(scored),
        "unsupported_labels": sum(1 for entry in scored if entry["human"] == 0),
        "unjudged": sum(1 for entry in scored if entry["judge"] is None),
        "exact_agreement": exact,
        "shipped": score(scored, SUPPORT_THRESHOLD),
        # Reported next to the shipped rule so the cost of tightening it is visible
        # rather than argued about, the way the screening threshold sweep does it.
        "strict": score(scored, 2),
        "by_pair": {
            pair: score([e for e in scored if e["pair"] == pair], SUPPORT_THRESHOLD)
            for pair in sorted({entry["pair"] for entry in scored})
        },
        "confusion": confusion(scored),
        "disagreements": [
            entry
            for entry in scored
            if kept(entry["judge"], SUPPORT_THRESHOLD) != (entry["human"] >= 1)
        ],
        "items": scored,
    }


def print_report(payload: dict) -> None:
    """Print the two headline rates and every disagreement worth reading."""

    shipped = payload["shipped"]
    print(f"\nscored {payload['scored']} labeled citations "
          f"({payload['unsupported_labels']} of them graded 0 by hand)")

    for name, entry in (("false drop", shipped["false_drop"]), ("catch", shipped["catch"])):
        if entry["rate"] is None:
            print(f"  {name:<12}: undefined — no labels in that class")
            continue
        low, high = entry["ci"]
        print(f"  {name:<12}: {entry['rate']:.1%} ({entry['n']} of {entry['of']}) "
              f"[{low:.0%}, {high:.0%}]")

    print(f"  exact grade : {payload['exact_agreement']['rate']:.1%}")
    print(f"  unjudged    : {payload['unjudged']}")

    print(f"\n{'human':>7}{'judge 2':>9}{'judge 1':>9}{'judge 0':>9}{'unjudged':>10}")
    for human, row in payload["confusion"].items():
        print(f"{human:>7}{row['2']:>9}{row['1']:>9}{row['0']:>9}{row['unjudged']:>10}")

    if payload["disagreements"]:
        print("\ndisagreements on keep-or-drop")
        for entry in payload["disagreements"]:
            print(f"  item {entry['item']:<4} human {entry['human']} "
                  f"judge {entry['judge']}  ({entry['pair']}) {entry['claim'][:60]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="score only the first N labels")
    args = parser.parse_args()

    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))["labels"]
    if args.limit:
        labels = labels[: args.limit]

    if not any(entry["grade"] == 0 for entry in labels):
        raise SystemExit(
            "no label is graded 0, so the catch rate cannot be measured and a judge "
            "that never rejects anything would score perfectly. Build and grade the "
            "judge set first: python -m evals.build.claim_support --judge-set"
        )

    scored = []
    for position, label in enumerate(labels, start=1):
        print(f"\rjudging {position}/{len(labels)}", end="", flush=True)
        scored.append(
            {
                "item": label["item"],
                "set": label["set"],
                "pair": label["pair"],
                "facet": label["facet"],
                "human": label["grade"],
                "judge": grade_one(label),
                "claim": label["claim"],
            }
        )
    print()

    payload = summarize(scored)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print_report(payload)
    print(f"\nwrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
