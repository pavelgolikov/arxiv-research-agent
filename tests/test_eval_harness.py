"""The guards that keep the published evaluation numbers honest.

The ablation's whole claim rests on pool depth equalling evaluation depth: every
chunk that can appear in a ranked list was judged, so nothing is scored as irrelevant
merely because nobody looked at it. Both runners enforce that at scoring time, and a
guard nobody tests is a guard that can quietly stop firing.
"""

import json

import pytest

from evals import build_index, run_retrieval


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
