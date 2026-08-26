"""Retry classification and the retry loop.

Getting this wrong is expensive in both directions: retrying a malformed PDF burns
model calls on something that will never succeed, and not retrying a 429 throws away
a paper for a reason that would have cleared on its own.
"""

import httpx
import pytest

from arxiv_reviewer.failures import (
    MAX_ATTEMPTS,
    PaperUnusableError,
    describe,
    is_retryable,
    with_retries,
)


def http_error(status_code: int) -> httpx.HTTPStatusError:
    """Build a status error the way httpx would raise one."""

    request = httpx.Request("GET", "https://arxiv.org/pdf/1234.5678")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


class TestIsRetryable:
    @pytest.mark.parametrize(
        "error",
        [
            httpx.TimeoutException("timed out"),
            httpx.ConnectError("refused"),
            httpx.ReadError("truncated"),
            httpx.RemoteProtocolError("bad framing"),
        ],
    )
    def test_transport_errors_are_transient(self, error):
        assert is_retryable(error) is True

    @pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
    def test_retryable_status_codes(self, status):
        assert is_retryable(http_error(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_client_errors_are_permanent(self, status):
        assert is_retryable(http_error(status)) is False

    @pytest.mark.parametrize(
        "code",
        ["RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL"],
    )
    def test_provider_codes_are_transient(self, code):
        assert is_retryable(RuntimeError(f"503 {code}: try again later")) is True

    def test_unusable_paper_is_never_retried(self):
        # A PDF that will not parse parses no better the second time.
        assert is_retryable(PaperUnusableError("not a PDF")) is False

    def test_unknown_errors_are_permanent(self):
        assert is_retryable(ValueError("schema violation")) is False


class TestWithRetries:
    def test_returns_immediately_on_success(self):
        calls = []
        result = with_retries(lambda: calls.append(1) or "done", sleep=lambda _s: None)
        assert result == "done"
        assert len(calls) == 1

    def test_retries_transient_failures_then_succeeds(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise httpx.TimeoutException("timed out")
            return "recovered"

        assert with_retries(flaky, sleep=lambda _s: None) == "recovered"
        assert len(attempts) == 3

    def test_gives_up_after_max_attempts(self):
        attempts = []

        def always_failing():
            attempts.append(1)
            raise httpx.ConnectError("refused")

        with pytest.raises(httpx.ConnectError):
            with_retries(always_failing, sleep=lambda _s: None)
        assert len(attempts) == MAX_ATTEMPTS

    def test_permanent_failure_is_not_retried(self):
        attempts = []

        def unusable():
            attempts.append(1)
            raise PaperUnusableError("not a PDF")

        with pytest.raises(PaperUnusableError):
            with_retries(unusable, sleep=lambda _s: None)
        assert len(attempts) == 1

    def test_backoff_grows_between_attempts(self):
        delays = []

        def always_failing():
            raise httpx.TimeoutException("timed out")

        with pytest.raises(httpx.TimeoutException):
            with_retries(
                always_failing,
                initial_interval=1.0,
                backoff_factor=2.0,
                jitter=False,
                sleep=delays.append,
            )
        assert delays == [1.0, 2.0]


class TestDescribe:
    def test_includes_the_error_type(self):
        assert describe(ValueError("bad input")).startswith("ValueError: ")

    def test_truncates_large_payloads(self):
        # Errors are stored in graph state and rendered into reports, so a megabyte
        # of provider response body must not end up in either.
        rendered = describe(RuntimeError("x" * 5000))
        assert len(rendered) <= len("RuntimeError: ") + 300
