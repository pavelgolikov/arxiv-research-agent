"""Sample citations for hand-judging whether an excerpt supports its claim.

The pipeline's three deterministic checks prove a citation is *referential*: the chunk
exists, belongs to the paper it is attributed to, and contains the quoted text. None of
them can prove the excerpt *supports* the claim built on it. That judgment needs a
reader, and it is the failure mode that survives every automated check.

An earlier 12-citation check gave 8 of 12, but that sample was drawn deliberately and
included two citations already flagged as weak, so the figure is biased downward and its
95% interval spans [39%, 86%]. This draws a uniform random sample with a fixed seed,
which is what a defensible rate needs.

The pipeline now runs an automated support judge over every citation, and these labels
are what it is scored against. Scoring needs failures to score on, and the uniform sample
contains none — 0 of 40 citations were graded `0`. A judge answering "supported" to
everything would score 100% against it. `--judge-set` therefore builds a second sheet
mixing unused real citations with constructed ones whose excerpt was swapped for another
quote from the same paper: right paper, wrong sentence, which is the failure mode a
reader actually meets. Which is which is kept out of the sheet and recorded in the key
file, so the grading stays blind.

    python -m evals.build.claim_support              # write the uniform sheet
    python -m evals.build.claim_support --judge-set  # write the judge evaluation sheet
    python -m evals.build.claim_support --collect    # parse both into labels/

Grades follow the convention used by the other datasets: `2` the excerpt establishes the
claim, `1` it supports part of it, `0` it does not support it. The published claim-support
rates are computed over the uniform sample alone, so the judge set cannot move them.
"""

import argparse
import json
import random
import re
from pathlib import Path

from arxiv_reviewer.analysis import SUPPORT_RUBRIC
from arxiv_reviewer.rag import DEFAULT_DATA_DIR
from arxiv_reviewer.workflow import persistent_graph, thread_config

from ..config import LABELS_DIR, RESULTS_DIR
from ..metrics import wilson_interval

SHEET_FILE = LABELS_DIR / "claim_support_sheet.md"
JUDGE_SHEET_FILE = LABELS_DIR / "claim_support_judge_sheet.md"
JUDGE_KEY_FILE = LABELS_DIR / "claim_support_judge_key.json"
LABELS_FILE = LABELS_DIR / "claim_support_labels.json"
GROUNDEDNESS_FILE = RESULTS_DIR / "groundedness.json"

SAMPLE_SIZE = 40
TOTAL_CITATIONS = 165
RANDOM_SEED = 20260825

# The judge set continues the uniform sheet's numbering, so item numbers stay unique
# across both files and a grade can never be attached to the wrong item.
JUDGE_SEED = 20260826
JUDGE_MISMATCHED = 20
JUDGE_REAL = 10

# One chunk can be cited by two different claims in the same facet, so the item index
# is the only unique key. Grading by (chunk_id, facet) would attach a grade to the
# wrong claim.
ITEM_BLOCK = re.compile(
    r"^### (?P<item>\d+)\.\s+`(?P<chunk_id>[^`]+)`\s+—\s+(?P<facet>\S+)\s*\n"
    r".*?- \*\*Claim:\*\*\s*(?P<claim>.+?)\n"
    r".*?- \*\*Excerpt:\*\*\s*“(?P<excerpt>.+?)”\s*\n"
    r".*?- \*\*Grade.*?:\*\*\s*`?(?P<grade>[012_])`?",
    re.MULTILINE | re.DOTALL,
)


def load_citations(thread_ids: list[str], data_dir: Path) -> list[dict]:
    """Read every validated citation out of the given runs, in a stable order."""

    graph = persistent_graph(data_dir)
    records = []

    for thread_id in thread_ids:
        values = graph.get_state(thread_config(thread_id)).values
        outcomes = sorted(
            values.get("analysis_outcomes", []), key=lambda item: item.search_position
        )
        for outcome in outcomes:
            if outcome.status != "ok" or outcome.analysis is None:
                continue
            for facet, claims in sorted(outcome.analysis.claims.items()):
                for claim in claims:
                    for evidence in claim.evidence:
                        records.append(
                            {
                                "thread_id": thread_id,
                                "arxiv_id": evidence.arxiv_id,
                                "facet": facet,
                                "chunk_id": evidence.chunk_id,
                                "page": evidence.page_number,
                                "claim": claim.text,
                                "excerpt": evidence.excerpt,
                            }
                        )

    return records


def draw(records: list[dict], size: int, seed: int) -> list[dict]:
    """Take a uniform random sample, keeping corpus order for a readable sheet."""

    generator = random.Random(seed)
    chosen = generator.sample(range(len(records)), min(size, len(records)))
    return [records[index] for index in sorted(chosen)]


def distribution(sample: list[dict], field: str) -> dict[str, int]:
    """Count how the sample fell across one field, so skew is visible."""

    counts: dict[str, int] = {}
    for record in sample:
        counts[record[field]] = counts.get(record[field], 0) + 1
    return dict(sorted(counts.items()))


def render_items(sample: list[dict], start: int) -> list[str]:
    """Render one graded item per citation, numbered from `start`."""

    lines = []
    for index, record in enumerate(sample, start=start):
        link = f"https://arxiv.org/pdf/{record['arxiv_id']}#page={record['page']}"
        lines += [
            f"### {index}. `{record['chunk_id']}` — {record['facet']}",
            "",
            f"- **Claim:** {record['claim']}",
            f"- **Excerpt:** “{record['excerpt']}”",
            f"- **Page:** [{record['arxiv_id']} p. {record['page']}]({link})",
            "- **Grade (2 / 1 / 0):** `_`",
            "",
        ]
    return lines


def instructions() -> list[str]:
    """Render the grading rubric every sheet shares with the automated judge.

    The rubric itself comes from `arxiv_reviewer.analysis`, so a human labeler and
    `judge_support` are answering one question. If they could drift apart, scoring the
    judge against these labels would measure the drift rather than the judge.
    """

    return [
        "For each item: read the excerpt, open the page if the excerpt alone is not "
        "enough, and grade how well the excerpt supports the claim.",
        "",
        *SUPPORT_RUBRIC.split("\n"),
        "",
        "Replace the `_` on each Grade line with a digit. Then run:",
        "",
        "```bash",
        "python -m evals.build.claim_support --collect",
        "```",
        "",
        "---",
        "",
    ]


def write_sheet(sample: list[dict], total: int) -> None:
    """Render the uniform sample's sheet to fill in."""

    lines = [
        "# Claim support labels",
        "",
        f"A uniform random sample of **{len(sample)} of {total}** citations from the "
        "committed groundedness runs.",
        "",
        *instructions(),
        *render_items(sample, start=1),
    ]

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    SHEET_FILE.write_text("\n".join(lines), encoding="utf-8")


def build_judge_set(
    records: list[dict],
    used: list[dict],
    mismatched: int,
    real: int,
    seed: int,
) -> list[dict]:
    """Draw the judge evaluation set: unused real citations plus swapped ones.

    A swap keeps a real claim and replaces its excerpt with a different quote from the
    same paper, then carries that quote's chunk, page, and facet so the item stays
    checkable — a labeler opening the link finds the excerpt on the page. Same paper
    rather than a different one on purpose: a quote from an unrelated paper is a
    negative nobody would ever produce, while right-paper-wrong-sentence is exactly
    what the deterministic checks let through.
    """

    generator = random.Random(seed)
    spent = {(item["chunk_id"], item["claim"]) for item in used}
    available = [
        record for record in records if (record["chunk_id"], record["claim"]) not in spent
    ]

    chosen = generator.sample(available, min(real + mismatched, len(available)))
    items = []

    for position, record in enumerate(chosen):
        if position < real:
            items.append({**record, "pair": "real"})
            continue

        alternatives = [
            other
            for other in records
            if other["arxiv_id"] == record["arxiv_id"]
            and other["claim"] != record["claim"]
            and other["excerpt"] != record["excerpt"]
        ]
        if not alternatives:
            # Nothing to swap with means no honest negative can be built here.
            items.append({**record, "pair": "real"})
            continue

        donor = generator.choice(alternatives)
        items.append(
            {
                **donor,
                "claim": record["claim"],
                "claim_facet": record["facet"],
                "pair": "mismatched",
            }
        )

    generator.shuffle(items)
    return items


def write_judge_sheet(items: list[dict], thread_ids: list[str], start: int) -> None:
    """Render the judge evaluation sheet and its answer key.

    The sheet does not say which items were swapped. A sheet of nothing but swaps, or
    one that marked them, would be graded `0` down the page without reading, and the
    labels would then record the construction rather than a judgment.
    """

    lines = [
        "# Claim support labels — judge evaluation set",
        "",
        f"**{len(items)}** further citations from the same runs, used to score the "
        "automated support judge in `evals/run_claim_judge.py`.",
        "",
        "Some are citations the pipeline really produced. In others the excerpt has "
        "been replaced by a different quote from the same paper. Which is which is "
        "recorded in `claim_support_judge_key.json` and deliberately not shown here — "
        "grade each item on what you read, exactly as in the first sheet.",
        "",
        *instructions(),
        *render_items(items, start=start),
    ]

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    JUDGE_SHEET_FILE.write_text("\n".join(lines), encoding="utf-8")

    key = {
        "seed": JUDGE_SEED,
        "source_threads": thread_ids,
        "first_item": start,
        "pairs": {
            str(start + offset): item["pair"] for offset, item in enumerate(items)
        },
    }
    JUDGE_KEY_FILE.write_text(
        json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )



def parse_sheet(path: Path, dataset: str) -> tuple[list[dict], list[int]]:
    """Parse one filled sheet into labels, reporting any item left ungraded."""

    labels: list[dict] = []
    ungraded: list[int] = []

    # Each block is parsed whole, so a malformed item is visible rather than skipped.
    for block in path.read_text(encoding="utf-8").split("\n### ")[1:]:
        match = ITEM_BLOCK.match("### " + block)
        if match is None:
            raise SystemExit(
                f"could not parse item starting: {block.splitlines()[0][:60]!r}. "
                "Keep the Claim, Excerpt, and Grade lines as generated."
            )

        item = int(match.group("item"))
        if match.group("grade") == "_":
            ungraded.append(item)
            continue

        labels.append(
            {
                "item": item,
                "set": dataset,
                "chunk_id": match.group("chunk_id"),
                "facet": match.group("facet"),
                # Claim and excerpt travel with the grade so this file alone is enough
                # to score a claim-support judge against.
                "claim": match.group("claim").strip(),
                "excerpt": match.group("excerpt").strip(),
                "grade": int(match.group("grade")),
            }
        )

    return labels, ungraded


def collect() -> dict:
    """Parse the filled sheets into the committed label file."""

    if not SHEET_FILE.exists():
        raise SystemExit(f"{SHEET_FILE} does not exist; generate the sheet first.")

    labels, ungraded = parse_sheet(SHEET_FILE, "uniform")
    for entry in labels:
        entry["pair"] = "real"

    if JUDGE_SHEET_FILE.exists():
        key = json.loads(JUDGE_KEY_FILE.read_text(encoding="utf-8"))["pairs"]
        judged, judge_ungraded = parse_sheet(JUDGE_SHEET_FILE, "judge")
        for entry in judged:
            # An item with no key entry would silently become a positive, so it is an
            # error rather than a default.
            pair = key.get(str(entry["item"]))
            if pair is None:
                raise SystemExit(
                    f"item {entry['item']} is not in {JUDGE_KEY_FILE.name}; "
                    "rebuild the judge sheet and its key together."
                )
            entry["pair"] = pair

        labels += judged
        ungraded += judge_ungraded

    if ungraded:
        raise SystemExit(
            f"{len(ungraded)} item(s) have no grade: {ungraded[:8]}. "
            "Every Grade line needs a digit before the labels can be written."
        )

    # The published rates describe the uniform sample only. The judge set is drawn to
    # exercise a judge, not to estimate a rate, and pooling it would bias every figure
    # in the README downward by exactly as many negatives as were constructed.
    uniform = [entry for entry in labels if entry["set"] == "uniform"]
    grades = [entry["grade"] for entry in uniform]
    total = len(grades)
    strict = grades.count(2)
    lenient = sum(1 for value in grades if value >= 1)

    strict_low, strict_high = wilson_interval(strict, total)
    lenient_low, lenient_high = wilson_interval(lenient, total)

    return {
        "question": "does the excerpt support the claim built on it",
        "grades": "2 establishes the claim, 1 supports it partly, 0 does not support it",
        "sample": {
            "size": total,
            "drawn_from": TOTAL_CITATIONS,
            "seed": RANDOM_SEED,
            "selection": "uniform random",
            "note": "published rates cover this sample only, not the judge set",
        },
        "judge_set": judge_set_summary(labels),
        "counts": {str(value): grades.count(value) for value in (2, 1, 0)},
        "strict_support_rate": strict / total if total else 0.0,
        "strict_ci": [strict_low, strict_high],
        "lenient_support_rate": lenient / total if total else 0.0,
        "lenient_ci": [lenient_low, lenient_high],
        "by_facet": {
            facet: {
                "n": sum(1 for e in uniform if e["facet"] == facet),
                "partial": sum(1 for e in uniform if e["facet"] == facet and e["grade"] == 1),
                "unsupported": sum(1 for e in uniform if e["facet"] == facet and e["grade"] == 0),
            }
            for facet in sorted({entry["facet"] for entry in uniform})
        },
        "labels": labels,
    }


def judge_set_summary(labels: list[dict]) -> dict:
    """Summarize what the judge will be scored against, including its negatives."""

    judged = [entry for entry in labels if entry["set"] == "judge"]
    by_pair = {}
    for pair in sorted({entry["pair"] for entry in judged}):
        grades = [entry["grade"] for entry in judged if entry["pair"] == pair]
        by_pair[pair] = {
            "n": len(grades),
            "counts": {str(value): grades.count(value) for value in (2, 1, 0)},
        }

    scored = [entry for entry in labels]
    return {
        "size": len(judged),
        "by_pair": by_pair,
        # What `run_claim_judge` actually scores against: every label, from both sheets.
        "scored_labels": len(scored),
        "unsupported_available": sum(1 for entry in scored if entry["grade"] == 0),
    }


def already_sampled(records: list[dict], size: int) -> list[dict]:
    """Return the citations the uniform sheet already spent.

    Read from the committed labels when they exist rather than re-drawn, so the judge
    set avoids the graded items even after the source runs have been regenerated and
    the seeded draw no longer reproduces them.
    """

    if LABELS_FILE.exists():
        payload = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
        return payload["labels"]
    return draw(records, size, RANDOM_SEED)


def source_threads(explicit: list[str] | None) -> list[str]:
    """Return the runs to sample from: the ones named, or the measured ones."""

    if explicit:
        return explicit
    groundedness = json.loads(GROUNDEDNESS_FILE.read_text(encoding="utf-8"))
    return [run["thread_id"] for run in groundedness["runs"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="parse the filled sheets")
    parser.add_argument(
        "--judge-set",
        action="store_true",
        help="write the judge evaluation sheet instead of the uniform one",
    )
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--mismatched", type=int, default=JUDGE_MISMATCHED)
    parser.add_argument("--real", type=int, default=JUDGE_REAL)
    parser.add_argument(
        "--thread-id",
        action="append",
        help="run to sample from (repeatable); defaults to the measured runs",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    if args.collect:
        payload = collect()
        LABELS_FILE.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        counts = payload["counts"]
        judge_set = payload["judge_set"]
        print(f"graded {payload['sample']['size']} citations (uniform sample)")
        print(f"  2 establishes  : {counts['2']}")
        print(f"  1 partly       : {counts['1']}")
        print(f"  0 no support   : {counts['0']}")
        print(f"\nstrict support rate  (2 only) : {payload['strict_support_rate']:.1%}")
        print(f"lenient support rate (1 or 2) : {payload['lenient_support_rate']:.1%}")
        print(f"\njudge set : {judge_set['size']} item(s), "
              f"{judge_set['scored_labels']} label(s) scoreable, "
              f"{judge_set['unsupported_available']} graded 0")
        if not judge_set["unsupported_available"]:
            print("  warning: no citation is graded 0, so a judge that never rejects "
                  "anything would score perfectly. Build the judge set first.")
        print(f"\nwrote {LABELS_FILE}")
        return

    thread_ids = source_threads(args.thread_id)
    records = load_citations(thread_ids, args.data_dir)

    if args.judge_set:
        used = already_sampled(records, args.size)
        items = build_judge_set(records, used, args.mismatched, args.real, JUDGE_SEED)
        start = max((entry["item"] for entry in used), default=0) + 1
        write_judge_sheet(items, thread_ids, start)

        pairs = distribution(items, "pair")
        print(f"citations available : {len(records)} across {len(thread_ids)} run(s)")
        print(f"already graded      : {len(used)}")
        print(f"judge set           : {len(items)} items numbered from {start}")
        print(f"  {pairs}  (per-item key in {JUDGE_KEY_FILE.name}, not in the sheet)")
        print(f"\nby paper : {distribution(items, 'arxiv_id')}")
        print(f"\nwrote {JUDGE_SHEET_FILE}")
        return

    sample = draw(records, args.size, RANDOM_SEED)
    write_sheet(sample, len(records))

    print(f"citations available : {len(records)} across {len(thread_ids)} run(s)")
    print(f"sampled             : {len(sample)} (uniform, seed {RANDOM_SEED})")
    print(f"\nby paper : {distribution(sample, 'arxiv_id')}")
    print(f"by facet : {distribution(sample, 'facet')}")
    print(f"\nwrote {SHEET_FILE}")


if __name__ == "__main__":
    main()
