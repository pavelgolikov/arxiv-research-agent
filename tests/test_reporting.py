"""Report rendering, including the fallback path that runs without a model."""

from arxiv_reviewer.reporting import (
    citation_link,
    escape_cell,
    failed_outcomes,
    render_markdown_fallback,
    run_status,
    selected_analyses,
)
from arxiv_reviewer.review_types import (
    AnalysisOutcome,
    EvidenceRef,
    GroundedAnalysis,
    ScreenOutcome,
    SupportedClaim,
)

from conftest import make_paper


def analysis(arxiv_id: str, dropped_claims: int = 0) -> GroundedAnalysis:
    claim = SupportedClaim(
        text="A supported finding",
        evidence=[
            EvidenceRef(
                chunk_id=f"{arxiv_id}:p2:c0",
                arxiv_id=arxiv_id,
                page_number=2,
                excerpt="quoted text",
            )
        ],
    )
    return GroundedAnalysis(
        arxiv_id=arxiv_id,
        title=f"Title {arxiv_id}",
        claims={"method": [claim], "main_findings": [claim]},
        dropped_claims=dropped_claims,
    )


def ok_outcome(arxiv_id: str, position: int) -> AnalysisOutcome:
    return AnalysisOutcome(
        arxiv_id=arxiv_id, search_position=position, status="ok",
        analysis=analysis(arxiv_id), chunk_count=10,
    )


def failed_outcome(arxiv_id: str, position: int) -> AnalysisOutcome:
    return AnalysisOutcome(
        arxiv_id=arxiv_id, search_position=position, status="failed",
        error="PaperUnusableError: not a PDF",
    )


def state_with(*outcomes, **extra):
    base = {
        "user_query": "a research question",
        "search_queries": ["query one"],
        "found_papers": [make_paper(o.arxiv_id) for o in outcomes],
        "selected_ids": [o.arxiv_id for o in outcomes],
        "analysis_outcomes": list(outcomes),
    }
    base.update(extra)
    return base


class TestRunStatus:
    def test_complete_when_everything_succeeded(self):
        assert run_status(state_with(ok_outcome("a", 0), ok_outcome("b", 1))) == "complete"

    def test_partial_when_a_branch_failed(self):
        assert run_status(state_with(ok_outcome("a", 0), failed_outcome("b", 1))) == "partial"

    def test_empty_when_nothing_was_analyzed(self):
        assert run_status(state_with()) == "empty"

    def test_partial_when_screening_failed(self):
        state = state_with(ok_outcome("a", 0))
        state["candidate_evaluations"] = [
            ScreenOutcome(arxiv_id="z", search_position=9, status="failed", error="boom")
        ]
        assert run_status(state) == "partial"


class TestOrdering:
    def test_analyses_come_back_in_search_order(self):
        # Reducers collect in completion order, so rendering must re-sort or the
        # report changes shape with concurrency.
        state = state_with(ok_outcome("c", 2), ok_outcome("a", 0), ok_outcome("b", 1))
        assert [a.arxiv_id for a in selected_analyses(state)] == ["a", "b", "c"]

    def test_failures_come_back_in_search_order(self):
        state = state_with(failed_outcome("c", 2), failed_outcome("a", 0))
        assert [o.arxiv_id for o in failed_outcomes(state)] == ["a", "c"]


class TestCellEscaping:
    def test_pipes_are_escaped(self):
        assert escape_cell("a | b") == "a \\| b"

    def test_newlines_become_spaces(self):
        assert "\n" not in escape_cell("line one\nline two")

    def test_citation_link_anchors_the_page(self):
        assert citation_link("2411.00750v2", 7) == (
            "[p. 7](https://arxiv.org/pdf/2411.00750v2#page=7)"
        )


class TestFallbackRenderer:
    def test_renders_without_a_model(self, fake_model):
        markdown = render_markdown_fallback(state_with(ok_outcome("a", 0)))
        assert "# Literature Review" in markdown
        assert fake_model.calls.get("synthesis") is None

    def test_reports_dropped_claims(self):
        outcome = AnalysisOutcome(
            arxiv_id="a", search_position=0, status="ok",
            analysis=analysis("a", dropped_claims=3),
        )
        assert "dropped 3 claim" in render_markdown_fallback(state_with(outcome))

    def test_lists_failures_so_they_cannot_be_omitted(self):
        markdown = render_markdown_fallback(
            state_with(ok_outcome("a", 0), failed_outcome("b", 1))
        )
        assert "## Failures" in markdown and "b:" in markdown

    def test_handles_a_run_with_no_papers(self):
        markdown = render_markdown_fallback(state_with())
        assert "No relevant papers were selected." in markdown
