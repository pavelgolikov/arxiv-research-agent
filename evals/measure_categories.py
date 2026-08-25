"""Would an arXiv category filter improve candidate quality?

A live run surfaced arXiv returning mostly off-topic papers for an on-topic query,
which suggested filtering the search by category. This measures that idea against the
labels that already exist instead of assuming it.

The frozen candidates were retrieved without a filter, so this answers "would a filter
have excluded the papers judged irrelevant, while keeping the ones judged central?" It
does not answer "what would a filtered search have returned instead" — that needs a
re-search, which would produce different candidates and void all 84 hand labels.

If the papers labeled irrelevant are cross-listed into the same categories as the
central ones, the filter cannot separate them and the idea is dead without writing it.
Categories are fetched once and cached, so the analysis re-runs offline.
"""

import argparse
import json
import time

import arxiv

from .config import CANDIDATES_FILE, RESULTS_DIR, SCREENING_LABELS

CATEGORIES_FILE = RESULTS_DIR / "candidate_categories.json"
RESULTS_FILE = RESULTS_DIR / "categories.json"

BATCH = 25
CENTRAL, RELATED = 2, 1

# Filters worth testing, narrowest first. `cs.*` stands for every computer-science
# category, which is what a blunt "restrict to CS" filter would do.
FILTERS = {
    "cs.CL": {"cs.CL"},
    "cs.CL + cs.LG": {"cs.CL", "cs.LG"},
    "cs.CL + cs.LG + cs.AI": {"cs.CL", "cs.LG", "cs.AI"},
    "cs.* (any CS)": None,
}


def fetch_categories(arxiv_ids: list[str]) -> dict[str, dict]:
    """Look up primary and cross-listed categories for each candidate."""

    client = arxiv.Client(page_size=BATCH, delay_seconds=3.0, num_retries=3)
    found: dict[str, dict] = {}

    for start in range(0, len(arxiv_ids), BATCH):
        batch = arxiv_ids[start : start + BATCH]
        if start:
            time.sleep(3.0)

        for result in client.results(arxiv.Search(id_list=batch, max_results=BATCH)):
            identifier = result.entry_id.rsplit("/", 1)[-1]
            found[identifier] = {
                "primary": result.primary_category,
                "categories": list(result.categories),
            }
        print(f"  fetched {min(start + BATCH, len(arxiv_ids))}/{len(arxiv_ids)}")

    return found


def load_categories(arxiv_ids: list[str], refresh: bool) -> dict[str, dict]:
    """Return cached categories, or fetch and cache them."""

    if not refresh and CATEGORIES_FILE.exists():
        cached = json.loads(CATEGORIES_FILE.read_text(encoding="utf-8"))
        if set(arxiv_ids) <= set(cached):
            print(f"categories: reusing {CATEGORIES_FILE.name}")
            return cached

    print(f"categories: fetching metadata for {len(arxiv_ids)} candidates")
    found = fetch_categories(arxiv_ids)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CATEGORIES_FILE.write_text(
        json.dumps(found, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {CATEGORIES_FILE}")
    return found


def matches(entry: dict, allowed: set[str] | None, primary_only: bool) -> bool:
    """Report whether a paper passes one filter definition."""

    fields = [entry["primary"]] if primary_only else entry["categories"]
    if allowed is None:
        return any(field.startswith("cs.") for field in fields)
    return any(field in allowed for field in fields)


def evaluate(rows: list[dict], allowed: set[str] | None, primary_only: bool) -> dict:
    """Count what one filter keeps and drops, by human label."""

    kept = [row for row in rows if matches(row["entry"], allowed, primary_only)]
    dropped = [row for row in rows if row not in kept]

    def count(subset: list[dict], label: int) -> int:
        return sum(1 for row in subset if row["label"] == label)

    central_kept = count(kept, CENTRAL)
    return {
        "kept": len(kept),
        "dropped": len(dropped),
        "central_kept": central_kept,
        "central_dropped": count(dropped, CENTRAL),
        "related_dropped": count(dropped, RELATED),
        "irrelevant_dropped": count(dropped, 0),
        "irrelevant_kept": count(kept, 0),
        "density_before": count(rows, CENTRAL) / len(rows),
        "density_after": central_kept / len(kept) if kept else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-fetch category metadata")
    args = parser.parse_args()

    frozen = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))
    labels = json.loads(SCREENING_LABELS.read_text(encoding="utf-8"))

    arxiv_ids = [
        candidate["arxiv_id"]
        for entry in frozen.values()
        for candidate in entry["candidates"]
    ]
    categories = load_categories(arxiv_ids, refresh=args.refresh)

    rows = []
    missing = []
    for query_id, entry in frozen.items():
        for candidate in entry["candidates"]:
            found = categories.get(candidate["arxiv_id"])
            if found is None:
                missing.append(candidate["arxiv_id"])
                continue
            rows.append(
                {
                    "query_id": query_id,
                    "arxiv_id": candidate["arxiv_id"],
                    "label": labels[query_id][candidate["arxiv_id"]],
                    "entry": found,
                }
            )

    if missing:
        print(f"\nno metadata for {len(missing)}: {missing[:5]}")

    results = {}
    for name, allowed in FILTERS.items():
        for primary_only in (False, True):
            scope = "primary only" if primary_only else "any category"
            results[f"{name} ({scope})"] = evaluate(rows, allowed, primary_only)

    payload = {
        "question": "would a category filter exclude irrelevant candidates and keep central ones",
        "candidates": len(rows),
        "caveat": (
            "Candidates were retrieved without a filter. This measures exclusion of what "
            "was returned, not what a filtered search would return instead."
        ),
        "filters": results,
        "category_counts_by_label": category_counts(rows),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print_report(payload)
    print(f"\nwrote {RESULTS_FILE}")


def category_counts(rows: list[dict]) -> dict:
    """Show which categories the papers of each label class actually live in."""

    names = {0: "irrelevant", RELATED: "related", CENTRAL: "central"}
    counts: dict[str, dict[str, int]] = {name: {} for name in names.values()}

    for row in rows:
        bucket = counts[names[row["label"]]]
        for category in row["entry"]["categories"]:
            bucket[category] = bucket.get(category, 0) + 1

    return {
        name: dict(sorted(bucket.items(), key=lambda kv: -kv[1])[:8])
        for name, bucket in counts.items()
    }


def print_report(payload: dict) -> None:
    """Print what each filter would have done."""

    print(f"\n{payload['candidates']} labeled candidates\n")
    print(f"{'filter':<38}{'kept':>6}{'irrel.drop':>12}{'central drop':>14}{'density':>10}")
    print("-" * 80)
    for name, stats in payload["filters"].items():
        print(
            f"{name:<38}{stats['kept']:>6}{stats['irrelevant_dropped']:>12}"
            f"{stats['central_dropped']:>14}"
            f"{stats['density_before']:>7.0%}->{stats['density_after']:.0%}"
        )

    print("\nwhere each label class actually lives")
    for name, counts in payload["category_counts_by_label"].items():
        top = ", ".join(f"{c} {n}" for c, n in list(counts.items())[:6])
        print(f"  {name:<11} {top}")


if __name__ == "__main__":
    main()
