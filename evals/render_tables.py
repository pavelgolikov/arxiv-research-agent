"""Render the README's evaluation tables from the committed results files.

The repository rule is that any number quoted anywhere comes from a committed results
file. This script makes that mechanical instead of a matter of discipline: it reads
`evals/results/*.json` and replaces the marked blocks in the README, so a table can
never drift away from the run that produced it.

Blocks are delimited by `<!-- eval:NAME -->` and `<!-- /eval:NAME -->`. Text outside
those markers is never touched, and a block is written to whichever target files
contain its markers — the README carries the headline tables, `DESIGN.md` the
methodology ones, and either may carry both.
"""

import argparse
import json
import re
from pathlib import Path

from arxiv_reviewer.rag import RETRIEVER_KINDS

from .config import EVALS_DIR, RESULTS_DIR

TARGETS = (EVALS_DIR.parent / "README.md", EVALS_DIR.parent / "DESIGN.md")

RETRIEVAL_FILE = RESULTS_DIR / "retrieval.json"
SCREENING_FILE = RESULTS_DIR / "screening.json"
GROUNDEDNESS_FILE = RESULTS_DIR / "groundedness.json"
COVERAGE_FILE = RESULTS_DIR / "index_coverage.json"


def load(path: Path) -> dict | None:
    """Read a results file, or return None when that runner has not been run."""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_table(data: dict) -> str:
    """Render the four-way ablation."""

    ndcg = f"ndcg@{data['config']['eval_k']}"
    recall = f"recall@{data['config']['recall_k']}"
    retrievers = data["retrievers"]

    lines = [
        "| Strategy | MRR | nDCG@10 | recall@5 (specific) | recall@5 (facet) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kind in RETRIEVER_KINDS:
        entry = retrievers.get(kind)
        if entry is None:
            continue
        specific = entry["by_kind"]["specific"]
        facet = entry["by_kind"]["facet"]
        lines.append(
            f"| `{kind}` | {entry['overall']['mrr']:.3f} | {entry['overall'][ndcg]:.3f} "
            f"| {specific[recall]:.3f} | {facet[recall]:.3f} |"
        )

    sample = next(iter(retrievers.values()))
    ceiling = f"{recall}_ceiling"
    lines.append(
        f"| _best achievable_ | — | — | _{sample['by_kind']['specific'][ceiling]:.3f}_ "
        f"| _{sample['by_kind']['facet'][ceiling]:.3f}_ |"
    )
    return "\n".join(lines)


def comparison_table(data: dict) -> str:
    """Render which differences survive resampling."""

    lines = [
        "| Comparison | Metric | Difference | 95% CI | Distinguishable |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in data["comparisons"]:
        scope = "" if row["questions"] == "all" else f" ({row['questions']})"
        verdict = "**yes**" if row["distinguishable"] else "no"
        lines.append(
            f"| `{row['baseline']}` → `{row['variant']}` | {row['metric']}{scope} "
            f"| {row['difference']:+.3f} "
            f"| [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | {verdict} |"
        )
    return "\n".join(lines)


def screening_table(data: dict) -> str:
    """Render the threshold sweep and the confusion grid."""

    current = data["config"]["current_default"]
    recommended = data["recommended_threshold"]

    lines = [
        "| Threshold | precision@4 (central) | precision@4 (related) | central recall "
        "| best achievable | queries under-filled |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for threshold in data["config"]["thresholds"]:
        pooled = data["sweep"][str(threshold)]["pooled"]
        note = []
        if threshold == current:
            note.append("current")
        if threshold == recommended:
            note.append("**recommended**")
        label = f"{threshold}" + (f" ({', '.join(note)})" if note else "")
        lines.append(
            f"| {label} | {pooled['precision_central']:.3f} "
            f"| {pooled['precision_related']:.3f} | {pooled['central_recall']:.3f} "
            f"| _{pooled['central_recall_ceiling']:.3f}_ "
            f"| {pooled['underfilled_queries']} |"
        )

    lines += [
        "",
        "| Model score | labeled irrelevant | labeled related | labeled central |",
        "| --- | --- | --- | --- |",
    ]
    for score, row in data["confusion"].items():
        lines.append(
            f"| {score} | {row['irrelevant']} | {row['related']} | {row['central']} |"
        )
    return "\n".join(lines)


def groundedness_table(data: dict) -> str:
    """Render the citation survival figures."""

    totals = data["totals"]
    proposed_claims = totals["claims_kept"] + totals["claims_dropped"]
    proposed_evidence = totals["evidence_kept"] + totals["evidence_dropped"]

    return "\n".join(
        [
            "| Measure | Value |",
            "| --- | --- |",
            f"| Papers analyzed | {totals['papers']} across {totals['runs']} runs |",
            f"| Claim-support rate | **{totals['claim_support_rate']:.1%}** "
            f"({totals['claims_kept']} of {proposed_claims} proposed claims kept) |",
            f"| Citation referential integrity | **{totals['citation_integrity']:.1%}** "
            f"({totals['evidence_kept']} of {proposed_evidence} proposed citations kept) |",
            f"| Citations per surviving claim | {totals['citations_per_claim']:.2f} |",
            f"| Independent re-validation | {totals['revalidation_rate']:.1%} "
            f"({totals['revalidated']} citations re-checked, "
            f"{totals['revalidation_failures']} failures) |",
        ]
    )


def coverage_line(data: dict) -> str:
    """Render the one-line statement that the pools still cover every ranked chunk."""

    return (
        f"Verified at pool depth {data['pool_depth']}: "
        f"{data['retrieved_chunks_checked']:,} retrieved chunks checked across "
        f"{data['questions']} questions, {len(data['unjudged_retrieved'])} unjudged."
    )


def render() -> dict[str, str]:
    """Build every block that has a results file behind it."""

    blocks: dict[str, str] = {}
    retrieval = load(RETRIEVAL_FILE)
    screening = load(SCREENING_FILE)
    groundedness = load(GROUNDEDNESS_FILE)
    coverage = load(COVERAGE_FILE)

    if retrieval:
        blocks["retrieval"] = retrieval_table(retrieval)
        blocks["comparisons"] = comparison_table(retrieval)
    if screening:
        blocks["screening"] = screening_table(screening)
    if groundedness:
        blocks["groundedness"] = groundedness_table(groundedness)
    if coverage:
        blocks["coverage"] = coverage_line(coverage)

    return blocks


def inject(text: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """Replace each marked block, leaving everything outside the markers alone."""

    replaced = []
    for name, body in blocks.items():
        # The body is matched non-greedily and may be empty, so a freshly added pair
        # of markers with nothing between them is filled in on the first run.
        pattern = re.compile(
            rf"(<!-- eval:{name} -->)(.*?)(<!-- /eval:{name} -->)",
            re.DOTALL,
        )
        text, count = pattern.subn(
            lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text
        )
        if count:
            replaced.append(name)

    return text, replaced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="update the marked blocks in README.md"
    )
    args = parser.parse_args()

    blocks = render()
    if not blocks:
        raise SystemExit("No results files found. Run the eval runners first.")

    if not args.write:
        for name, body in blocks.items():
            print(f"\n<!-- eval:{name} -->\n{body}\n<!-- /eval:{name} -->")
        print("\n(pass --write to update README.md in place)")
        return

    placed: set[str] = set()
    changed = False

    for target in TARGETS:
        if not target.exists():
            continue

        original = target.read_text(encoding="utf-8")
        updated, replaced = inject(original, blocks)
        placed.update(replaced)

        if updated == original:
            print(f"{target.name} already up to date.")
            continue

        target.write_text(updated, encoding="utf-8")
        changed = True
        print(f"updated {target.name} ({', '.join(sorted(replaced))})")

    # A block with no marker anywhere would silently go unpublished, which is exactly
    # the drift these markers exist to prevent.
    orphaned = sorted(set(blocks) - placed)
    if orphaned:
        print(f"\nno marker in any target for: {', '.join(orphaned)}")

    if not changed and not orphaned:
        print("all generated tables match their results files.")


if __name__ == "__main__":
    main()
