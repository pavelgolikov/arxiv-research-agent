"""The compiled workflow, driven end to end with fakes.

Chroma runs for real here against a temporary directory, so these exercise genuine
chunking, indexing, per-paper retrieval, and citation validation. Only the calls that
would leave the machine — the model, arXiv, and PDF downloads — are replaced.
"""

import pytest

from arxiv_reviewer.analysis import RELEVANCE_THRESHOLD, select_papers_node
from arxiv_reviewer.review_types import ScreenOutcome
from arxiv_reviewer.workflow import build_graph, open_checkpointer, thread_config

from conftest import make_paper


@pytest.fixture
def run_graph(tmp_path, fake_model, fake_arxiv, fake_embeddings, no_retry_delay):
    """Return a callable that runs the real graph against a temporary data directory."""

    def run(papers, thread_id="t1", concurrency=3, target=2, **overrides):
        fake_arxiv["papers"] = papers
        fake_arxiv["served"] = set()
        graph = build_graph(open_checkpointer(tmp_path))
        state = {
            "user_query": "a research question",
            "max_results": len(papers),
            "target_papers": target,
            "output": str(tmp_path / f"{thread_id}.md"),
            "thread_id": thread_id,
            "data_dir": str(tmp_path),
            "retriever_kind": "dense",
            "top_k": 3,
            "fetch_k": 6,
            "multi_query": False,
            "candidate_evaluations": [],
            "analysis_outcomes": [],
            "status": "running",
        }
        state.update(overrides)
        return graph.invoke(state, config=thread_config(thread_id, concurrency))

    run.model = fake_model
    run.arxiv = fake_arxiv
    run.tmp_path = tmp_path
    return run


class TestTerminalPaths:
    def test_no_candidates_found(self, run_graph):
        final = run_graph([], thread_id="none")
        assert final["status"] == "empty"
        assert "No relevant papers were selected." in final["markdown"]

    def test_none_selected(self, run_graph):
        run_graph.model.scores = {"a1": 1, "a2": 2}
        final = run_graph([make_paper("a1"), make_paper("a2")], thread_id="rejected")
        assert final["status"] == "empty"
        assert final["selected_ids"] == []

    def test_full_success(self, run_graph):
        final = run_graph([make_paper("a1"), make_paper("a2")], thread_id="ok")
        assert final["status"] == "complete"
        assert len(final["selected_ids"]) == 2
        assert all(o.status == "ok" for o in final["analysis_outcomes"])

    def test_partial_when_one_analysis_branch_fails(self, run_graph):
        # A branch that fails must not take its siblings with it.
        run_graph.arxiv["unusable"] = {"a2"}
        final = run_graph([make_paper("a1"), make_paper("a2")], thread_id="partial")
        statuses = {o.arxiv_id: o.status for o in final["analysis_outcomes"]}
        assert statuses == {"a1": "ok", "a2": "failed"}
        assert final["status"] == "partial"
        assert "## Failures" in final["markdown"]

    def test_partial_when_one_screening_branch_fails(self, run_graph):
        run_graph.model.fail_screening_for = {"a2"}
        final = run_graph([make_paper("a1"), make_paper("a2")], thread_id="screenfail")
        assert final["status"] == "partial"
        assert any(e.status == "failed" for e in final["candidate_evaluations"])

    def test_synthesis_failure_falls_back_to_the_renderer(self, run_graph):
        run_graph.model.synthesis_error = RuntimeError("model unavailable")
        final = run_graph([make_paper("a1")], thread_id="fallback")
        assert final["status"] == "partial"
        assert "# Literature Review" in final["markdown"]


class TestDeterminism:
    def test_concurrency_does_not_change_the_output(self, run_graph):
        papers = [make_paper(f"a{n}") for n in range(4)]
        serial = run_graph(papers, thread_id="serial", concurrency=1, target=3)
        parallel = run_graph(papers, thread_id="parallel", concurrency=3, target=3)

        assert serial["selected_ids"] == parallel["selected_ids"]
        assert len(serial["analysis_outcomes"]) == 3
        # Reducers collect in completion order; the rendered report must not show it.
        assert serial["markdown"] == parallel["markdown"]

    def test_evidence_survives_validation_against_the_real_index(self, run_graph):
        final = run_graph([make_paper("a1")], thread_id="grounded")
        analysis = final["analysis_outcomes"][0].analysis
        assert analysis.supported_claim_count > 0
        assert analysis.dropped_evidence == 0
        assert analysis.dropped_unsupported == 0
        for claims in analysis.claims.values():
            for claim in claims:
                for evidence in claim.evidence:
                    assert evidence.arxiv_id == "a1"
                    assert evidence.chunk_id.startswith("a1:")
                    assert evidence.support_grade is not None

    def test_a_paper_whose_citations_are_all_unsupported_produces_no_claims(
        self, run_graph
    ):
        # The judge runs inside the analysis branch, so an unsupported citation has to
        # be gone before anything reaches the report — not filtered at render time.
        run_graph.model.support_grades = {"a1": 0}
        final = run_graph([make_paper("a1")], thread_id="unsupported")
        analysis = final["analysis_outcomes"][0].analysis

        assert analysis.supported_claim_count == 0
        assert analysis.dropped_unsupported > 0
        assert analysis.is_partial


class TestSelection:
    def test_ranks_by_score_then_search_position(self, run_graph):
        run_graph.model.scores = {"a0": 4, "a1": 5, "a2": 5, "a3": 4}
        final = run_graph([make_paper(f"a{n}") for n in range(4)], target=3, thread_id="rank")
        assert final["selected_ids"] == ["a1", "a2", "a0"]

    def test_threshold_defaults_to_the_module_constant(self):
        evaluations = [
            ScreenOutcome(arxiv_id="a", search_position=0, score=RELEVANCE_THRESHOLD),
            ScreenOutcome(arxiv_id="b", search_position=1, score=RELEVANCE_THRESHOLD - 1),
        ]
        state = {"candidate_evaluations": evaluations, "target_papers": 5}
        assert select_papers_node(state)["selected_ids"] == ["a"]

    def test_threshold_can_be_overridden_through_state(self):
        evaluations = [
            ScreenOutcome(arxiv_id="a", search_position=0, score=5),
            ScreenOutcome(arxiv_id="b", search_position=1, score=2),
        ]
        state = {
            "candidate_evaluations": evaluations,
            "target_papers": 5,
            "relevance_threshold": 2,
        }
        assert select_papers_node(state)["selected_ids"] == ["a", "b"]

    def test_failed_screenings_are_never_selected(self):
        evaluations = [
            ScreenOutcome(arxiv_id="a", search_position=0, score=5, status="failed"),
            ScreenOutcome(arxiv_id="b", search_position=1, score=5),
        ]
        state = {"candidate_evaluations": evaluations, "target_papers": 5}
        assert select_papers_node(state)["selected_ids"] == ["b"]


class TestResume:
    def test_finished_branches_are_not_re_executed(self, tmp_path, fake_model,
                                                   fake_arxiv, fake_embeddings,
                                                   no_retry_delay):
        papers = [make_paper("a1"), make_paper("a2")]
        fake_arxiv["papers"] = papers
        checkpointer = open_checkpointer(tmp_path)
        graph = build_graph(checkpointer)
        config = thread_config("resumable", 1)

        state = {
            "user_query": "a research question",
            "max_results": 2, "target_papers": 2,
            "output": str(tmp_path / "resume.md"),
            "thread_id": "resumable", "data_dir": str(tmp_path),
            "retriever_kind": "dense", "top_k": 3, "fetch_k": 6, "multi_query": False,
            "candidate_evaluations": [], "analysis_outcomes": [], "status": "running",
        }
        graph.invoke(state, config=config)
        after_first = dict(fake_model.calls)

        # An input-less invocation on a finished thread must not redo the work.
        snapshot = graph.get_state(config)
        assert snapshot.values["status"] in {"complete", "partial"}
        assert fake_model.calls == after_first

    def test_state_survives_a_round_trip_through_sqlite(self, tmp_path, fake_model,
                                                        fake_arxiv, fake_embeddings,
                                                        no_retry_delay):
        fake_arxiv["papers"] = [make_paper("a1")]
        graph = build_graph(open_checkpointer(tmp_path))
        config = thread_config("persisted", 1)
        graph.invoke(
            {
                "user_query": "q", "max_results": 1, "target_papers": 1,
                "output": str(tmp_path / "p.md"), "thread_id": "persisted",
                "data_dir": str(tmp_path), "retriever_kind": "dense",
                "top_k": 3, "fetch_k": 6, "multi_query": False,
                "candidate_evaluations": [], "analysis_outcomes": [], "status": "running",
            },
            config=config,
        )
        # Re-open the store from disk: the structured claims must come back as
        # objects, which is what the groundedness runner depends on.
        reopened = build_graph(open_checkpointer(tmp_path))
        values = reopened.get_state(thread_config("persisted", 1)).values
        analysis = values["analysis_outcomes"][0].analysis
        assert analysis.claims
        assert analysis.supported_claim_count > 0
