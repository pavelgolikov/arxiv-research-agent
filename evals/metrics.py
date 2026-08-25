"""Ranking metrics for graded relevance judgments.

Grades are `0` irrelevant, `1` partial, `2` directly answers. The binary metrics
treat any positive grade as relevant; `ndcg_at` uses the grades themselves, which
is the reason the labels are graded at all — a binary nDCG collapses to a much
blunter measure.

These functions take a ranked list of chunk identifiers and a mapping of the chunks
judged relevant for that question. They read no files and call no models, so they
can be checked against cases whose answers are known by hand.
"""

import math
import random

RELEVANT_GRADE = 1


def reciprocal_rank(ranked: list[str], relevant: dict[str, int]) -> float:
    """Return 1/rank of the first relevant item, or 0.0 if none was retrieved."""

    for position, chunk_id in enumerate(ranked, start=1):
        if relevant.get(chunk_id, 0) >= RELEVANT_GRADE:
            return 1.0 / position
    return 0.0


def recall_at(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Return the fraction of relevant chunks appearing in the top k."""

    if not relevant:
        return 0.0

    found = sum(
        1 for chunk_id in ranked[:k] if relevant.get(chunk_id, 0) >= RELEVANT_GRADE
    )
    return found / len(relevant)


def recall_ceiling(relevant: dict[str, int], k: int) -> float:
    """Return the best recall@k reachable when k is smaller than the relevant set.

    Reported next to recall everywhere it appears. Without it a reader cannot tell a
    retriever that ranked badly from a question with more answers than the cutoff
    has slots.
    """

    if not relevant:
        return 0.0
    return min(k, len(relevant)) / len(relevant)


def dcg(grades: list[int]) -> float:
    """Return discounted cumulative gain for grades already in rank order."""

    return sum(
        (2**grade - 1) / math.log2(position + 2)
        for position, grade in enumerate(grades)
    )


def ndcg_at(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Return normalized DCG at k against the ideal ordering of the judged grades."""

    ideal = dcg(sorted(relevant.values(), reverse=True)[:k])
    if not ideal:
        return 0.0

    return dcg([relevant.get(chunk_id, 0) for chunk_id in ranked[:k]]) / ideal


def mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""

    return sum(values) / len(values) if values else 0.0


def paired_bootstrap(
    differences: list[float],
    iterations: int = 20000,
    seed: int = 20260825,
) -> tuple[float, float, float]:
    """Return the mean paired difference and its 95% confidence interval.

    Fifty questions is a small benchmark, and two retrievers can differ by a few
    points of nDCG purely by chance. Resampling the per-question differences says
    whether an observed gap survives that noise. An interval spanning zero means the
    benchmark cannot distinguish the two configurations, which is a result worth
    reporting rather than a gap worth quoting.

    The seed is fixed so a rerun reproduces the interval exactly.
    """

    if not differences:
        return 0.0, 0.0, 0.0

    generator = random.Random(seed)
    size = len(differences)
    means = sorted(
        sum(generator.choices(differences, k=size)) / size
        for _ in range(iterations)
    )
    observed = sum(differences) / size
    return observed, means[int(0.025 * iterations)], means[int(0.975 * iterations)]
