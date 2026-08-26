"""The guards that keep the published evaluation numbers honest.

The ablation's whole claim rests on pool depth equalling evaluation depth: every
chunk that can appear in a ranked list was judged, so nothing is scored as irrelevant
merely because nobody looked at it. Both runners enforce that at scoring time, and a
guard nobody tests is a guard that can quietly stop firing.
"""

import json

import pytest

from arxiv_reviewer.analysis import SUPPORT_RUBRIC, SUPPORT_THRESHOLD
from evals import build_index, run_claim_judge, run_groundedness, run_retrieval
from evals.build import claim_support


class TestCoverageGuard:
    def test_run_retrieval_accepts_a_fully_judged_ranking(self):
        run_retrieval.check_coverage("q1", ["c1", "c2"], {"c1", "c2", "c3"})

    def test_run_retrieval_rejects_an_unjudged_chunk(self):
        with pytest.raises(SystemExit) as failure:
            run_retrieval.check_coverage("q1", ["c1", "unjudged"], {"c1"})
        assert "unjudged" in str(failure.value)

    def test_the_error_says_not_to_widen_the_cutoff(self):
        # The tempting fix is to lower k until the error stops. The message exists to
        # say that would hide the problem rather than solve it.
        with pytest.raises(SystemExit) as failure:
            run_retrieval.check_coverage("q1", ["nope"], set())
        assert "Rebuild the pools" in str(failure.value)

    def test_empty_ranking_is_trivially_covered(self):
        run_retrieval.check_coverage("q1", [], {"c1"})


class TestEvaluationDepth:
    def test_eval_depth_matches_pool_depth(self):
        # If these drift apart the labels no longer cover what is scored, which is
        # the one failure mode the whole dataset design guards against.
        from evals.config import POOL_DEPTH

        assert run_retrieval.EVAL_K == POOL_DEPTH
        assert run_retrieval.FETCH_K == max(POOL_DEPTH * 2, 20)

    def test_recall_cutoff_is_within_the_judged_depth(self):
        assert run_retrieval.RECALL_K <= run_retrieval.EVAL_K


class TestCommittedResults:
    """The published numbers must stay internally consistent."""

    def test_no_unjudged_chunk_was_scored(self):
        coverage = json.loads(
            (build_index.COVERAGE_FILE).read_text(encoding="utf-8")
        )
        assert coverage["complete"] is True
        assert coverage["unjudged_retrieved"] == []

    def test_recall_never_exceeds_its_ceiling(self):
        results = json.loads(
            (run_retrieval.RESULTS_FILE).read_text(encoding="utf-8")
        )
        for entry in results["retrievers"].values():
            for group in entry["by_kind"].values():
                assert group["recall@5"] <= group["recall@5_ceiling"] + 1e-9

    def test_retrievers_are_not_all_identical(self):
        # Identical scores across four strategies would mean the strategy name is not
        # reaching the retriever, not that the strategies are equally good.
        results = json.loads(
            (run_retrieval.RESULTS_FILE).read_text(encoding="utf-8")
        )
        scores = {
            round(entry["overall"]["ndcg@10"], 4)
            for entry in results["retrievers"].values()
        }
        assert len(scores) > 1


def judged(*pairs: tuple[int, int | None], pair: str = "real") -> list[dict]:
    """Build scored items from (human grade, judge grade) pairs."""

    return [
        {"item": index, "set": "judge", "pair": pair, "facet": "method",
         "human": human, "judge": judge, "claim": "a claim"}
        for index, (human, judge) in enumerate(pairs, start=1)
    ]


class TestClaimJudgeScoring:
    """The two rates that decide whether the support check is worth running.

    They fail in opposite directions, so a single "accuracy" would hide either one.
    These pin the arithmetic against tables whose answers are known by hand.
    """

    def test_catch_rate_counts_only_the_citations_a_reader_rejected(self):
        # Two rejected by hand, one of which the judge also rejects.
        result = run_claim_judge.score(judged((0, 0), (0, 2), (2, 2)), threshold=1)
        assert result["catch"]["n"] == 1 and result["catch"]["of"] == 2

    def test_false_drop_rate_counts_only_the_citations_a_reader_kept(self):
        result = run_claim_judge.score(judged((2, 0), (1, 1), (0, 0)), threshold=1)
        assert result["false_drop"]["n"] == 1 and result["false_drop"]["of"] == 2

    def test_partial_grades_count_as_kept_at_the_shipped_threshold(self):
        result = run_claim_judge.score(judged((1, 1)), threshold=SUPPORT_THRESHOLD)
        assert result["false_drop"]["n"] == 0

    def test_the_strict_threshold_treats_a_partial_as_a_drop(self):
        result = run_claim_judge.score(judged((1, 1)), threshold=2)
        assert result["false_drop"]["n"] == 1

    def test_an_unjudged_citation_counts_as_a_drop(self):
        # `apply_support_judge` discards it, so the measured rate has to as well or the
        # eval would describe a kinder rule than the one that ships.
        assert run_claim_judge.kept(None, 1) is False
        result = run_claim_judge.score(judged((2, None)), threshold=1)
        assert result["false_drop"]["n"] == 1

    def test_a_class_with_no_labels_reports_no_rate_rather_than_zero(self):
        # Before the judge set existed there were no 0-graded labels at all. Reporting
        # 0.0 there would read as "catches nothing" instead of "nothing to catch".
        result = run_claim_judge.score(judged((2, 2)), threshold=1)
        assert result["catch"]["rate"] is None

    def test_confusion_grid_totals_the_scored_items(self):
        grid = run_claim_judge.confusion(judged((2, 2), (1, 0), (0, None)))
        assert grid["2"]["2"] == 1
        assert grid["1"]["0"] == 1
        assert grid["0"]["unjudged"] == 1


class TestClaimSupportLabels:
    def test_the_sheet_and_the_judge_grade_one_rubric(self):
        # Scoring the judge against these labels only means something while both are
        # answering the same question, so the sheet renders the pipeline's own rubric.
        assert SUPPORT_RUBRIC.split("\n")[0] in claim_support.instructions()

    def test_published_rates_ignore_the_judge_set(self):
        # The judge set is drawn to exercise a judge, not to estimate a rate. Pooling
        # its constructed negatives would drag every published figure downward.
        payload = json.loads(
            claim_support.LABELS_FILE.read_text(encoding="utf-8")
        )
        uniform = [entry for entry in payload["labels"] if entry["set"] == "uniform"]
        assert payload["sample"]["size"] == len(uniform)
        assert payload["counts"]["2"] + payload["counts"]["1"] + payload["counts"]["0"] == len(uniform)

    def test_every_label_records_which_set_and_pair_it_came_from(self):
        payload = json.loads(
            claim_support.LABELS_FILE.read_text(encoding="utf-8")
        )
        for entry in payload["labels"]:
            assert entry["set"] in {"uniform", "judge"}
            assert entry["pair"] in {"real", "mismatched"}


class TestGroundednessStages:
    """Citations are rejected at two stages; the rates must not blame the wrong one."""

    def test_a_judge_rejection_does_not_lower_referential_integrity(self):
        # The judge only ever sees citations that already resolved, so moving a
        # rejection from one stage to the other must not change the referential rate.
        without = run_groundedness.integrity(9, 2, 0, judged=9)
        with_judge = run_groundedness.integrity(8, 2, 1, judged=8)
        assert without["citation_integrity"] == with_judge["citation_integrity"]

    def test_support_integrity_is_measured_against_what_resolved(self):
        result = run_groundedness.integrity(8, 2, 1, judged=8)
        assert result["support_integrity"] == 8 / 9

    def test_overall_integrity_counts_both_stages(self):
        result = run_groundedness.integrity(8, 2, 1, judged=8)
        assert result["overall_integrity"] == 8 / 11

    def test_a_run_predating_the_judge_reports_no_support_rate(self):
        # 100% would read as "the judge accepted everything" rather than "the judge
        # never ran", which is the difference between a result and a blank.
        assert run_groundedness.integrity(9, 2, 0, judged=0)["support_integrity"] is None
