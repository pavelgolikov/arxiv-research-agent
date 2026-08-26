"""Ranking metrics, checked against cases whose answers are known by hand.

Every published evaluation number rests on these functions, so they are tested in
isolation: no files, no network, no model.
"""

from evals.metrics import (
    dcg,
    mean,
    ndcg_at,
    paired_bootstrap,
    recall_at,
    recall_ceiling,
    reciprocal_rank,
)

# Two chunks that fully answer the question, one that partially answers it.
RELEVANT = {"a": 2, "b": 1, "c": 2}


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "x", "y"], RELEVANT) == 1.0

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "b"], RELEVANT) == 1 / 3

    def test_partial_grade_counts_as_relevant(self):
        assert reciprocal_rank(["b"], RELEVANT) == 1.0

    def test_nothing_relevant_retrieved(self):
        assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0

    def test_empty_ranking(self):
        assert reciprocal_rank([], RELEVANT) == 0.0


class TestRecall:
    def test_finds_two_of_three(self):
        assert recall_at(["a", "b", "x"], RELEVANT, 5) == 2 / 3

    def test_cutoff_truncates_the_ranking(self):
        assert recall_at(["a", "b", "c"], RELEVANT, 1) == 1 / 3

    def test_no_relevant_set(self):
        assert recall_at(["a"], {}, 5) == 0.0

    def test_ceiling_when_cutoff_is_smaller_than_relevant_set(self):
        # The reason recall is always reported beside its ceiling: three answers
        # cannot fit in one slot, so 1/3 is a perfect score here, not a poor one.
        assert recall_ceiling(RELEVANT, 1) == 1 / 3

    def test_ceiling_is_one_when_everything_fits(self):
        assert recall_ceiling(RELEVANT, 5) == 1.0

    def test_recall_never_exceeds_its_ceiling(self):
        for k in (1, 2, 3, 5, 10):
            assert recall_at(["a", "b", "c"], RELEVANT, k) <= recall_ceiling(RELEVANT, k)


class TestNDCG:
    def test_ideal_order_scores_one(self):
        assert ndcg_at(["a", "c", "b"], RELEVANT, 10) == 1.0

    def test_worse_order_scores_less(self):
        assert ndcg_at(["b", "a", "c"], RELEVANT, 10) < 1.0

    def test_grade_two_outranks_grade_one(self):
        # Graded labels exist for this: a binary nDCG could not tell these apart.
        assert ndcg_at(["a", "b"], RELEVANT, 2) > ndcg_at(["b", "a"], RELEVANT, 2)

    def test_cutoff_excludes_later_hits(self):
        assert ndcg_at(["x", "x", "a"], RELEVANT, 2) == 0.0

    def test_nothing_relevant_retrieved(self):
        assert ndcg_at(["x", "y"], RELEVANT, 10) == 0.0

    def test_no_relevant_set(self):
        assert ndcg_at(["a"], {}, 10) == 0.0

    def test_zero_grades_contribute_nothing(self):
        assert dcg([0, 0, 0]) == 0.0


class TestPairedBootstrap:
    def test_constant_difference_gives_a_tight_interval(self):
        observed, low, high = paired_bootstrap([0.1] * 30)
        assert observed == 0.1
        assert low == high == 0.1

    def test_zero_mean_difference_spans_zero(self):
        # The case that keeps the ablation honest: no real difference must produce an
        # interval containing zero, so the comparison is reported as indistinguishable.
        _observed, low, high = paired_bootstrap([0.5, -0.5] * 25)
        assert low < 0.0 < high

    def test_large_consistent_difference_excludes_zero(self):
        _observed, low, high = paired_bootstrap([0.4, 0.5, 0.6] * 10)
        assert low > 0.0

    def test_is_reproducible(self):
        differences = [0.3, -0.1, 0.2, 0.05] * 10
        assert paired_bootstrap(differences) == paired_bootstrap(differences)

    def test_empty_input(self):
        assert paired_bootstrap([]) == (0.0, 0.0, 0.0)


class TestMean:
    def test_empty(self):
        assert mean([]) == 0.0

    def test_average(self):
        assert mean([1.0, 0.0]) == 0.5
