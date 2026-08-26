"""The model client's contract with the rest of the pipeline.

`include_raw` is what makes token counts reachable, and it also stops LangChain from
raising when a response violates the schema. Every caller depends on that exception to
turn bad model output into a recorded failure, so the re-raise is asserted here. Nothing
else covers it: the other suites replace these helpers with fakes.
"""

import threading

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage

from arxiv_reviewer.gemini_client import (
    generate_structured,
    record_tokens,
    reset_usage,
    usage_totals,
)
from arxiv_reviewer.review_types import SearchPlan


class FakeStructuredModel:
    """Stand in for the chain `with_structured_output(..., include_raw=True)` builds."""

    def __init__(self, result: dict):
        self.result = result
        self.include_raw = None

    def with_structured_output(self, result_type, include_raw=False):
        self.include_raw = include_raw
        return self

    def invoke(self, prompt):
        return self.result


@pytest.fixture
def clean_usage():
    reset_usage()
    yield
    reset_usage()


class TestStructuredOutput:
    def build(self, monkeypatch, result):
        model = FakeStructuredModel(result)
        monkeypatch.setattr("arxiv_reviewer.gemini_client.gemini_llm", lambda: model)
        return model

    def test_a_schema_violation_is_raised_not_swallowed(self, monkeypatch, clean_usage):
        # LangChain returns the parse failure under `parsing_error` rather than raising
        # it. Callers record a typed failure from the exception, so it has to come back.
        self.build(
            monkeypatch,
            {
                "raw": AIMessage(content="not a plan"),
                "parsed": None,
                "parsing_error": OutputParserException("queries field missing"),
            },
        )

        with pytest.raises(OutputParserException, match="queries field missing"):
            generate_structured("plan a search", SearchPlan)

    def test_a_failed_call_records_no_tokens(self, monkeypatch, clean_usage):
        self.build(
            monkeypatch,
            {
                "raw": AIMessage(content="not a plan"),
                "parsed": None,
                "parsing_error": OutputParserException("bad"),
            },
        )

        with pytest.raises(OutputParserException):
            generate_structured("plan a search", SearchPlan)
        assert usage_totals()["calls"] == 0

    def test_a_good_response_is_parsed_and_counted(self, monkeypatch, clean_usage):
        model = self.build(
            monkeypatch,
            {
                "raw": AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
                ),
                "parsed": SearchPlan(queries=["transformer interpretability"]),
                "parsing_error": None,
            },
        )

        plan = generate_structured("plan a search", SearchPlan)

        assert plan.queries == ["transformer interpretability"]
        assert model.include_raw is True
        assert usage_totals() == {"calls": 1, "input_tokens": 120, "output_tokens": 30}


class TestUsageAccounting:
    def test_reset_clears_what_was_recorded(self, clean_usage):
        record_tokens({"input_tokens": 5, "output_tokens": 2})
        reset_usage()
        assert usage_totals() == {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def test_a_call_without_metadata_is_ignored(self, clean_usage):
        record_tokens(None)
        record_tokens({})
        assert usage_totals()["calls"] == 0

    def test_counts_survive_concurrent_branches(self, clean_usage):
        # Screening and analysis fan out across threads, so every branch appends to the
        # same list. A lost update here would understate the measured cost.
        def work():
            for _ in range(200):
                record_tokens({"input_tokens": 1, "output_tokens": 1})

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert usage_totals() == {"calls": 800, "input_tokens": 800, "output_tokens": 800}
