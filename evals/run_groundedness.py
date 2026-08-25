"""Measure how reliably generated reports ground their claims in real evidence.

Reads completed runs out of the checkpoint database rather than parsing the Markdown.
The rendered report keeps only page links, discarding the chunk identifiers, but
`GroundedAnalysis`, `SupportedClaim`, and `EvidenceRef` survive in the checkpoint
because those types are registered in `CHECKPOINT_TYPES`.

The framing matters. Citations that reach a finished report are valid by
construction, because `validate_claim` already discarded the invalid ones during
analysis. Measuring referential integrity on the finished output would return 100%
every time and demonstrate nothing. What is worth measuring is the survival rate of
what the model originally proposed, which `GroundedAnalysis` records in its
`dropped_claims` and `dropped_evidence` counters.

Re-validation is the exception, and it is a regression check rather than a finding.
It re-runs the same deterministic checks against the stored chunks and is expected to
return 100%. A lower number means the index drifted away from the run it belongs to,
or that validation was not applied.
"""

import argparse
import json
from pathlib import Path

from arxiv_reviewer.analysis import normalize
from arxiv_reviewer.rag import DEFAULT_DATA_DIR, load_chunks
from arxiv_reviewer.workflow import persistent_graph, thread_config

from .config import RESULTS_DIR

RESULTS_FILE = RESULTS_DIR / "groundedness.json"


def ratio(kept: int, dropped: int) -> float:
    """Return the share that survived, or 0.0 when nothing was proposed."""

    total = kept + dropped
    return kept / total if total else 0.0


def revalidate(analysis, thread_id: str, data_dir: Path) -> dict:
    """Re-run the three citation checks against the chunks stored for this run."""

    chunks = {
        document.metadata["chunk_id"]: document
        for document in load_chunks(thread_id, arxiv_id=analysis.arxiv_id, data_dir=data_dir)
    }

    checked = 0
    failures: list[dict] = []

    for facet, claims in analysis.claims.items():
        for claim in claims:
            for evidence in claim.evidence:
                checked += 1
                document = chunks.get(evidence.chunk_id)

                if document is None:
                    reason = "chunk_id not in the index"
                elif document.metadata["arxiv_id"] != analysis.arxiv_id:
                    reason = "chunk belongs to a different paper"
                elif document.metadata["page_number"] != evidence.page_number:
                    reason = "page number does not match the chunk"
                elif normalize(evidence.excerpt) not in normalize(document.page_content):
                    reason = "excerpt not found in the cited chunk"
                else:
                    continue

                failures.append(
                    {
                        "arxiv_id": analysis.arxiv_id,
                        "facet": facet,
                        "chunk_id": evidence.chunk_id,
                        "reason": reason,
                    }
                )

    return {"checked": checked, "failures": failures}


def measure_paper(analysis, thread_id: str, data_dir: Path) -> dict:
    """Compute survival and re-validation figures for one analyzed paper."""

    kept_claims = analysis.supported_claim_count
    kept_evidence = sum(
        len(claim.evidence) for claims in analysis.claims.values() for claim in claims
    )
    checks = revalidate(analysis, thread_id, data_dir)

    return {
        "arxiv_id": analysis.arxiv_id,
        "title": analysis.title,
        "claims_kept": kept_claims,
        "claims_dropped": analysis.dropped_claims,
        "evidence_kept": kept_evidence,
        "evidence_dropped": analysis.dropped_evidence,
        "claim_support_rate": ratio(kept_claims, analysis.dropped_claims),
        "citation_integrity": ratio(kept_evidence, analysis.dropped_evidence),
        "revalidated": checks["checked"],
        "revalidation_failures": checks["failures"],
        # Drops are accumulated per paper, not per facet, so only the surviving
        # counts can be attributed to a facet.
        "claims_by_facet": {
            facet: len(claims) for facet, claims in sorted(analysis.claims.items())
        },
    }


def measure_run(thread_id: str, data_dir: Path) -> dict:
    """Read one thread from its checkpoint and measure every analyzed paper."""

    snapshot = persistent_graph(data_dir).get_state(thread_config(thread_id))
    if snapshot.created_at is None:
        raise SystemExit(f"Unknown thread: {thread_id}")

    values = snapshot.values
    papers = [
        measure_paper(outcome.analysis, thread_id, data_dir)
        for outcome in sorted(
            values.get("analysis_outcomes", []), key=lambda item: item.search_position
        )
        if outcome.status == "ok" and outcome.analysis is not None
    ]

    return {
        "thread_id": thread_id,
        "user_query": values.get("user_query", ""),
        "status": values.get("status", ""),
        "retriever_kind": values.get("retriever_kind", ""),
        "output": values.get("output", ""),
        "papers": papers,
    }


def totals(runs: list[dict]) -> dict:
    """Pool the per-paper counts across every run."""

    papers = [paper for run in runs for paper in run["papers"]]
    field = lambda name: sum(paper[name] for paper in papers)

    kept_claims, dropped_claims = field("claims_kept"), field("claims_dropped")
    kept_evidence, dropped_evidence = field("evidence_kept"), field("evidence_dropped")
    revalidated = field("revalidated")
    failures = [f for paper in papers for f in paper["revalidation_failures"]]

    by_facet: dict[str, int] = {}
    for paper in papers:
        for facet, count in paper["claims_by_facet"].items():
            by_facet[facet] = by_facet.get(facet, 0) + count

    return {
        "runs": len(runs),
        "papers": len(papers),
        "claims_kept": kept_claims,
        "claims_dropped": dropped_claims,
        "claim_support_rate": ratio(kept_claims, dropped_claims),
        "evidence_kept": kept_evidence,
        "evidence_dropped": dropped_evidence,
        "citation_integrity": ratio(kept_evidence, dropped_evidence),
        "citations_per_claim": kept_evidence / kept_claims if kept_claims else 0.0,
        "revalidated": revalidated,
        "revalidation_failures": len(failures),
        "revalidation_rate": (revalidated - len(failures)) / revalidated if revalidated else 0.0,
        "claims_by_facet": dict(sorted(by_facet.items())),
    }


def print_report(payload: dict) -> None:
    """Print the pooled figures and the per-paper breakdown."""

    pooled = payload["totals"]
    print(f"\n{payload['config']['note']}\n")
    print(f"  papers analyzed        : {pooled['papers']} across {pooled['runs']} run(s)")
    print(f"  claims kept / proposed : {pooled['claims_kept']} / "
          f"{pooled['claims_kept'] + pooled['claims_dropped']}"
          f"   -> claim-support rate {pooled['claim_support_rate']:.1%}")
    print(f"  citations kept / prop. : {pooled['evidence_kept']} / "
          f"{pooled['evidence_kept'] + pooled['evidence_dropped']}"
          f"   -> citation integrity {pooled['citation_integrity']:.1%}")
    print(f"  citations per claim    : {pooled['citations_per_claim']:.2f}")
    print(f"  re-validated           : {pooled['revalidated']} citations, "
          f"{pooled['revalidation_failures']} failure(s) -> {pooled['revalidation_rate']:.1%}")

    print(f"\n{'paper':<16}{'claims':>9}{'dropped':>9}{'cites':>8}{'dropped':>9}{'integrity':>11}")
    print("-" * 62)
    for run in payload["runs"]:
        for paper in run["papers"]:
            print(
                f"{paper['arxiv_id']:<16}{paper['claims_kept']:>9}{paper['claims_dropped']:>9}"
                f"{paper['evidence_kept']:>8}{paper['evidence_dropped']:>9}"
                f"{paper['citation_integrity']:>11.1%}"
            )

    print("\nsurviving claims by facet")
    for facet, count in pooled["claims_by_facet"].items():
        print(f"  {facet:<22}{count:>4}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thread-id", action="append", required=True, help="run to measure (repeatable)"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    runs = [measure_run(thread_id, args.data_dir) for thread_id in args.thread_id]
    payload = {
        "config": {
            "source": "langgraph checkpoint state, not the rendered Markdown",
            "note": (
                "Rates measure what survived validation out of what the model proposed. "
                "Citations in a finished report are valid by construction, so measuring "
                "the report itself would trivially return 100%."
            ),
            "revalidation": "deterministic re-check of every committed citation",
        },
        "totals": totals(runs),
        "runs": runs,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print_report(payload)
    print(f"\nwrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
