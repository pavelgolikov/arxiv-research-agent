"""Sample citations for hand-judging whether an excerpt supports its claim.

The pipeline's three deterministic checks prove a citation is *referential*: the chunk
exists, belongs to the paper it is attributed to, and contains the quoted text. None of
them can prove the excerpt *supports* the claim built on it. That judgment needs a
reader, and it is the failure mode that survives every automated check.

An earlier 12-citation check gave 8 of 12, but that sample was drawn deliberately and
included two citations already flagged as weak, so the figure is biased downward and its
95% interval spans [39%, 86%]. This draws a uniform random sample with a fixed seed,
which is what a defensible rate needs.

    python -m evals.build.claim_support              # write the sheet
    python -m evals.build.claim_support --collect    # parse it into labels/

Grades follow the convention used by the other datasets: `2` the excerpt establishes the
claim, `1` it supports part of it, `0` it does not support it.
"""

import argparse
import json
import random
import re
from pathlib import Path

from arxiv_reviewer.rag import DEFAULT_DATA_DIR
from arxiv_reviewer.workflow import persistent_graph, thread_config

from ..config import LABELS_DIR, RESULTS_DIR
from ..metrics import wilson_interval

SHEET_FILE = LABELS_DIR / "claim_support_sheet.md"
LABELS_FILE = LABELS_DIR / "claim_support_labels.json"
GROUNDEDNESS_FILE = RESULTS_DIR / "groundedness.json"

SAMPLE_SIZE = 40
TOTAL_CITATIONS = 165
RANDOM_SEED = 20260825

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


def write_sheet(sample: list[dict], total: int) -> None:
    """Render the sheet to fill in."""

    lines = [
        "# Claim support labels",
        "",
        f"A uniform random sample of **{len(sample)} of {total}** citations from the "
        "committed groundedness runs.",
        "",
        "For each item: read the excerpt, open the page if the excerpt alone is not "
        "enough, and grade how well the excerpt supports the claim.",
        "",
        "- `2` — the excerpt establishes the claim.",
        "- `1` — the excerpt supports part of the claim, or supports it with a "
        "qualifier the quote does not carry.",
        "- `0` — the excerpt does not support the claim.",
        "",
        "Judge the **excerpt against the claim**, not whether the claim is true of the "
        "paper. A correct statement quoted from the wrong sentence is still a `0`.",
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

    for index, record in enumerate(sample, start=1):
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

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    SHEET_FILE.write_text("\n".join(lines), encoding="utf-8")


def collect() -> dict:
    """Parse the filled sheet into the committed label file."""

    if not SHEET_FILE.exists():
        raise SystemExit(f"{SHEET_FILE} does not exist; generate the sheet first.")

    text = SHEET_FILE.read_text(encoding="utf-8")

    labels: list[dict] = []
    ungraded: list[int] = []

    # Each block is parsed whole, so a malformed item is visible rather than skipped.
    for block in text.split("\n### ")[1:]:
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
                "chunk_id": match.group("chunk_id"),
                "facet": match.group("facet"),
                # Claim and excerpt travel with the grade so this file alone is enough
                # to score a claim-support judge against.
                "claim": match.group("claim").strip(),
                "excerpt": match.group("excerpt").strip(),
                "grade": int(match.group("grade")),
            }
        )

    if ungraded:
        raise SystemExit(
            f"{len(ungraded)} item(s) have no grade: {ungraded[:8]}. "
            "Every Grade line needs a digit before the labels can be written."
        )

    grades = [entry["grade"] for entry in labels]
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
        },
        "counts": {str(value): grades.count(value) for value in (2, 1, 0)},
        "strict_support_rate": strict / total if total else 0.0,
        "strict_ci": [strict_low, strict_high],
        "lenient_support_rate": lenient / total if total else 0.0,
        "lenient_ci": [lenient_low, lenient_high],
        "by_facet": {
            facet: {
                "n": sum(1 for e in labels if e["facet"] == facet),
                "partial": sum(1 for e in labels if e["facet"] == facet and e["grade"] == 1),
                "unsupported": sum(1 for e in labels if e["facet"] == facet and e["grade"] == 0),
            }
            for facet in sorted({entry["facet"] for entry in labels})
        },
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="parse the filled sheet")
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    if args.collect:
        payload = collect()
        LABELS_FILE.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        counts = payload["counts"]
        print(f"graded {payload['sample']['size']} citations")
        print(f"  2 establishes  : {counts['2']}")
        print(f"  1 partly       : {counts['1']}")
        print(f"  0 no support   : {counts['0']}")
        print(f"\nstrict support rate  (2 only) : {payload['strict_support_rate']:.1%}")
        print(f"lenient support rate (1 or 2) : {payload['lenient_support_rate']:.1%}")
        print(f"\nwrote {LABELS_FILE}")
        return

    groundedness = json.loads(GROUNDEDNESS_FILE.read_text(encoding="utf-8"))
    thread_ids = [run["thread_id"] for run in groundedness["runs"]]

    records = load_citations(thread_ids, args.data_dir)
    sample = draw(records, args.size, RANDOM_SEED)
    write_sheet(sample, len(records))

    print(f"citations available : {len(records)} across {len(thread_ids)} run(s)")
    print(f"sampled             : {len(sample)} (uniform, seed {RANDOM_SEED})")
    print(f"\nby paper : {distribution(sample, 'arxiv_id')}")
    print(f"by facet : {distribution(sample, 'facet')}")
    print(f"\nwrote {SHEET_FILE}")


if __name__ == "__main__":
    main()
