"""Failure classification and retrying shared by the parallel branches."""

import random
import time
from typing import Callable, TypeVar

import httpx

T = TypeVar("T")

MAX_ATTEMPTS = 3
INITIAL_INTERVAL = 1.0
BACKOFF_FACTOR = 2.0

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
RETRYABLE_TRANSPORT = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)
RETRYABLE_PROVIDER_CODES = (
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
)


class PaperUnusableError(Exception):
    """Raised when a paper cannot be used and retrying would not help."""


def is_retryable(error: BaseException) -> bool:
    """Report whether an error is transient enough to be worth retrying."""

    if isinstance(error, PaperUnusableError):
        return False
    if isinstance(error, RETRYABLE_TRANSPORT):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS

    message = str(error).upper()
    return any(code in message for code in RETRYABLE_PROVIDER_CODES)


def describe(error: BaseException) -> str:
    """Render an error for storage without leaking large payloads."""

    return f"{type(error).__name__}: {str(error)[:300]}"


def with_retries(
    operation: Callable[[], T],
    max_attempts: int = MAX_ATTEMPTS,
    initial_interval: float = INITIAL_INTERVAL,
    backoff_factor: float = BACKOFF_FACTOR,
    jitter: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry an operation while its failures look transient.

    Parallel branches retry here rather than through LangGraph's node-level
    `RetryPolicy` because a `RetryPolicy` failure aborts the whole run. Doing it
    inside the branch lets one paper fail without discarding its siblings.
    """

    delay = initial_interval

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == max_attempts or not is_retryable(error):
                raise
            sleep(delay * (1 + random.random() * 0.1) if jitter else delay)
            delay *= backoff_factor

    raise AssertionError("unreachable")
