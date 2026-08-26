"""Shared fixtures. The suite runs with no network access and no API key.

Every model, arXiv, and HTTP call is replaced by a fake. Chroma is *not* faked: it
runs for real against a temporary directory with deterministic embeddings, so the
graph tests exercise genuine chunking, indexing, and retrieval and only the calls
that would leave the machine are stubbed.

The fakes are content-addressed rather than a queue of scripted replies. Screening
and analysis fan out with `Send`, so call order depends on scheduling; a fake that
popped answers in sequence would pass or fail depending on concurrency. Deriving each
reply from its own prompt keeps the tests deterministic at any `max_concurrency`.
"""

import os
import re
import socket
import threading

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from arxiv_reviewer.review_types import (
    DraftClaim,
    DraftEvidence,
    FacetDraft,
    PaperMetadata,
    ParsedPage,
    ParsedPaper,
    RelevanceDecision,
    SearchPlan,
    SupportVerdict,
    SupportVerdicts,
)

# Chroma's telemetry client would otherwise try to phone home and trip the socket guard.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

EMBEDDING_SIZE = 32
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}

# Matches one rendered block from `analysis.format_context`.
CONTEXT_BLOCK = re.compile(r"\[([^\]\s]+)\] \(page (\d+)\)\n(.+?)(?=\n\n\[|\Z)", re.DOTALL)

# Matches one numbered item from `analysis.judge_support`.
JUDGE_ITEM = re.compile(r"^(\d+)\.\nClaim: (.+)$", re.MULTILINE)

# What the fake judge answers unless a test says otherwise, so every existing graph
# test keeps the citations it always had.
FULL_SUPPORT = 2


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any outbound connection so a missing fake surfaces as a test error."""

    real_connect = socket.socket.connect

    def guarded(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host in LOCAL_HOSTS:
            return real_connect(self, address, *args, **kwargs)
        raise RuntimeError(
            f"the test suite must not reach the network (attempted {address!r}); "
            "a model, arXiv, or HTTP call is missing its fake"
        )

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    """Run as if no API key were configured, the way CI would."""

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Give Chroma deterministic vectors instead of calling the embedding API."""

    embeddings = DeterministicFakeEmbedding(size=EMBEDDING_SIZE)
    monkeypatch.setattr("arxiv_reviewer.rag.get_embeddings", lambda: embeddings)
    return embeddings


def make_paper(arxiv_id: str, title: str = "", **overrides) -> PaperMetadata:
    """Build paper metadata without repeating every required field."""

    fields = {
        "arxiv_id": arxiv_id,
        "title": title or f"Paper {arxiv_id}",
        "authors": ["A. Author", "B. Author"],
        "abstract": f"Abstract for {arxiv_id}.",
        "published": "2026-01-01",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "entry_url": f"https://arxiv.org/abs/{arxiv_id}",
    }
    fields.update(overrides)
    return PaperMetadata(**fields)


def make_parsed(arxiv_id: str, pages: int = 3) -> ParsedPaper:
    """Build a parsed paper whose text is long enough to chunk into several pieces."""

    body = [
        ParsedPage(
            page_number=number,
            text=(
                f"Page {number} of {arxiv_id}. "
                + " ".join(
                    f"Sentence {index} on page {number} discusses the method in detail."
                    for index in range(30)
                )
            ),
        )
        for number in range(1, pages + 1)
    ]
    return ParsedPaper(arxiv_id=arxiv_id, pages=body, page_count=pages)


class FakeModel:
    """Stand-in for the Gemini helpers, answering from the prompt it was given."""

    def __init__(self, scores: dict[str, int] | None = None):
        self.scores = scores or {}
        # Maps a substring of the claim to the grade the judge should return for it.
        # A `None` grade omits the verdict entirely, standing in for a reply with a
        # hole in it. Keyed by content rather than call order, like the other fakes.
        self.support_grades: dict[str, int | None] = {}
        self.calls: dict[str, int] = {}
        self.seen_papers: list[str] = []
        self.fail_screening_for: set[str] = set()
        self.fail_facets_for: set[str] = set()
        self.synthesis_error: Exception | None = None
        self._lock = threading.Lock()

    def _record(self, name: str) -> None:
        with self._lock:
            self.calls[name] = self.calls.get(name, 0) + 1

    def structured(self, prompt: str, result_type):
        """Answer a structured request based on what the prompt asks for."""

        if result_type is SearchPlan:
            self._record("search_plan")
            return SearchPlan(queries=["query one", "query two"])

        if result_type is RelevanceDecision:
            return self._screen(prompt)

        if result_type is FacetDraft:
            return self._analyze(prompt)

        if result_type is SupportVerdicts:
            return self._judge(prompt)

        raise AssertionError(f"unexpected structured request for {result_type!r}")

    def _screen(self, prompt: str) -> RelevanceDecision:
        self._record("screen")
        match = re.search(r"arXiv ID: (\S+)", prompt)
        arxiv_id = match.group(1) if match else "unknown"

        with self._lock:
            self.seen_papers.append(arxiv_id)
        if arxiv_id in self.fail_screening_for:
            raise RuntimeError(f"screening failed for {arxiv_id}")

        score = self.scores.get(arxiv_id, 5)
        return RelevanceDecision(
            arxiv_id=arxiv_id,
            is_relevant=score >= 4,
            score=score,
            reason=f"scored {score}",
        )

    def _analyze(self, prompt: str) -> FacetDraft:
        """Cite the first shown chunk, quoting it verbatim so validation passes."""

        self._record("facet")
        blocks = CONTEXT_BLOCK.findall(prompt)
        if not blocks:
            return FacetDraft(claims=[])

        chunk_id, _page, text = blocks[0]
        if chunk_id.split(":")[0] in self.fail_facets_for:
            raise RuntimeError(f"analysis failed for {chunk_id}")

        return FacetDraft(
            claims=[
                DraftClaim(
                    text=f"A finding drawn from {chunk_id}",
                    evidence=[
                        DraftEvidence(chunk_id=chunk_id, excerpt=text.strip()[:120])
                    ],
                )
            ]
        )

    def _judge(self, prompt: str) -> SupportVerdicts:
        """Grade every numbered item, supporting the claim fully unless told not to."""

        self._record("support_judge")
        verdicts = []

        for index, claim in JUDGE_ITEM.findall(prompt):
            grade = next(
                (value for key, value in self.support_grades.items() if key in claim),
                FULL_SUPPORT,
            )
            if grade is None:
                continue
            verdicts.append(
                SupportVerdict(index=int(index), grade=grade, reason="fake verdict")
            )

        return SupportVerdicts(verdicts=verdicts)

    def text(self, prompt: str) -> str:
        """Stand in for synthesis."""

        self._record("synthesis")
        if self.synthesis_error is not None:
            raise self.synthesis_error
        return "# Synthesized Review\n\nWritten by the fake model.\n"


@pytest.fixture
def fake_model(monkeypatch):
    """Patch every module that imported a model helper into its own namespace."""

    model = FakeModel()
    monkeypatch.setattr("arxiv_reviewer.analysis.generate_structured", model.structured)
    monkeypatch.setattr("arxiv_reviewer.retrieval.generate_structured", model.structured)
    monkeypatch.setattr("arxiv_reviewer.reporting.generate_text", model.text)
    return model


@pytest.fixture
def fake_arxiv(monkeypatch):
    """Return frozen search results and parsed pages instead of touching arXiv."""

    state = {"papers": [], "unusable": set(), "served": set()}

    def search(query: str, max_results: int):
        """Serve papers not yet returned, so distinct queries surface distinct results.

        `search_node` divides the result budget across the planned queries, so a fake
        that returned the same prefix every time would starve every query after the
        first and the graph would see fewer candidates than the test asked for.
        """

        fresh = [
            paper for paper in state["papers"] if paper.arxiv_id not in state["served"]
        ][:max_results]
        state["served"].update(paper.arxiv_id for paper in fresh)
        return fresh

    def fetch(paper: PaperMetadata) -> ParsedPaper:
        if paper.arxiv_id in state["unusable"]:
            from arxiv_reviewer.failures import PaperUnusableError

            raise PaperUnusableError(f"{paper.arxiv_id} could not be parsed")
        return make_parsed(paper.arxiv_id)

    monkeypatch.setattr("arxiv_reviewer.retrieval.search_arxiv", search)
    monkeypatch.setattr("arxiv_reviewer.analysis.fetch_parsed_paper", fetch)
    return state


@pytest.fixture
def no_retry_delay(monkeypatch):
    """Remove backoff sleeps so failure paths do not slow the suite down."""

    monkeypatch.setattr("arxiv_reviewer.failures.INITIAL_INTERVAL", 0.0)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
