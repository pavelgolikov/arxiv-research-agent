"""Citation validation and chunking.

Validation runs in two layers and these tests cover both.

`validate_claim` is the anti-hallucination guarantee: a claim survives only if it
cites a chunk that was actually shown, belonging to the paper under analysis, quoting
text that genuinely occurs there. These tests pin both directions — faithful citations
must survive, and every way of being unfaithful must not.

`apply_support_judge` answers what no deterministic check can: whether the verified
quote supports the sentence built on it. Its tests pin the drop rule, and pin that a
citation cannot reach the report through a gap in the judge's reply.
"""

import pytest
from langchain_core.documents import Document

from arxiv_reviewer import analysis
from arxiv_reviewer.analysis import (
    apply_support_judge,
    EVIDENCE_EXCERPT_CHARS,
    judge_support,
    normalize,
    SUPPORT_THRESHOLD,
    validate_claim,
)
from arxiv_reviewer.rag import build_chunk_id, chunk_pages
from arxiv_reviewer.review_types import (
    DraftClaim,
    DraftEvidence,
    EvidenceRef,
    ParsedPage,
    SupportedClaim,
    SupportVerdict,
    SupportVerdicts,
)

PAPER = "2411.00750v2"
CHUNK_TEXT = (
    "Models tend to over-sample easy queries and under-sample the ones they have "
    "yet to master, which narrows the tail over successive iterations."
)


def chunk(chunk_id: str = "c1", arxiv_id: str = PAPER, text: str = CHUNK_TEXT) -> Document:
    return Document(
        page_content=text,
        metadata={"arxiv_id": arxiv_id, "page_number": 4, "chunk_id": chunk_id},
    )


def claim_citing(chunk_id: str, excerpt: str) -> DraftClaim:
    return DraftClaim(
        text="A claim about sampling",
        evidence=[DraftEvidence(chunk_id=chunk_id, excerpt=excerpt)],
    )


def supported_claim(*excerpts: str, text: str = "A claim about sampling") -> SupportedClaim:
    """Build a claim that has already passed the three deterministic checks."""

    return SupportedClaim(
        text=text,
        evidence=[
            EvidenceRef(
                chunk_id=f"c{index}",
                arxiv_id=PAPER,
                page_number=4,
                excerpt=excerpt,
            )
            for index, excerpt in enumerate(excerpts, start=1)
        ],
    )


def grading(*grades: int | None):
    """Return a `judge_support` stand-in answering these grades in order."""

    return lambda items: list(grades)


class TestValidateClaim:
    def test_verbatim_excerpt_survives(self):
        shown = {"c1": chunk()}
        result, dropped = validate_claim(
            claim_citing("c1", "over-sample easy queries"), PAPER, shown
        )
        assert result is not None and dropped == 0
        assert result.evidence[0].page_number == 4

    def test_hallucinated_chunk_id_is_dropped(self):
        result, dropped = validate_claim(
            claim_citing("c99", "over-sample easy queries"), PAPER, {"c1": chunk()}
        )
        assert result is None and dropped == 1

    def test_chunk_from_another_paper_is_dropped(self):
        shown = {"c1": chunk(arxiv_id="9999.11111v1")}
        result, dropped = validate_claim(
            claim_citing("c1", "over-sample easy queries"), PAPER, shown
        )
        assert result is None and dropped == 1

    def test_paraphrase_is_dropped(self):
        result, _dropped = validate_claim(
            claim_citing("c1", "the model prefers simple questions"), PAPER, {"c1": chunk()}
        )
        assert result is None

    def test_fabricated_text_is_dropped(self):
        result, _dropped = validate_claim(
            claim_citing("c1", "trained on 40 billion synthetic dialogues"),
            PAPER,
            {"c1": chunk()},
        )
        assert result is None

    def test_empty_excerpt_is_dropped(self):
        result, dropped = validate_claim(claim_citing("c1", "   "), PAPER, {"c1": chunk()})
        assert result is None and dropped == 1

    def test_claim_keeps_valid_citation_and_drops_the_bad_one(self):
        mixed = DraftClaim(
            text="A claim",
            evidence=[
                DraftEvidence(chunk_id="c1", excerpt="over-sample easy queries"),
                DraftEvidence(chunk_id="c99", excerpt="invented"),
            ],
        )
        result, dropped = validate_claim(mixed, PAPER, {"c1": chunk()})
        assert result is not None
        assert len(result.evidence) == 1 and dropped == 1

    def test_excerpt_is_truncated_to_the_limit(self):
        long_text = "word " * 400
        shown = {"c1": chunk(text=long_text)}
        result, _dropped = validate_claim(claim_citing("c1", long_text), PAPER, shown)
        assert result is not None
        assert len(result.evidence[0].excerpt) <= EVIDENCE_EXCERPT_CHARS


class TestHyphenationRegression:
    """PDF line-break hyphenation once discarded half of every run's valid work.

    Extraction preserves the hyphen a typesetter inserted at a line break, so a chunk
    reads "self- improvement" where the paper reads "self-improvement". A model
    quoting it faithfully was rejected, and the resulting 48.6% "citation integrity"
    measured the PDF extractor rather than the model.
    """

    def test_excerpt_matches_across_a_line_break_hyphen(self):
        shown = {"c1": chunk(text="guided self- improvement rebalances the sampling")}
        result, dropped = validate_claim(
            claim_citing("c1", "guided self-improvement rebalances"), PAPER, shown
        )
        assert result is not None and dropped == 0

    def test_quote_drops_the_line_break_hyphen_entirely(self):
        # The original discovered case: the chunk reads "lead- ing" and the model,
        # reading it as prose, quotes "leading".
        shown = {"c1": chunk(text="this imbalance is exacerbated, lead- ing to a long tail")}
        result, dropped = validate_claim(
            claim_citing("c1", "exacerbated, leading to a long tail"), PAPER, shown
        )
        assert result is not None and dropped == 0

    def test_hyphen_absent_from_the_source_is_still_rejected(self):
        # The fold has to stay narrow. A model can only turn "fine tuning" into
        # "fine-tuning" by changing what it was shown, which is not a verbatim quote.
        shown = {"c1": chunk(text="we apply fine tuning to the base model")}
        result, _dropped = validate_claim(
            claim_citing("c1", "fine-tuning to the base"), PAPER, shown
        )
        assert result is None

    def test_normalize_folds_case_whitespace_and_hyphenation(self):
        assert normalize("Lead-  ing\n Edge") == normalize("leading edge")

    def test_folding_does_not_accept_unrelated_text(self):
        # The fold must stay narrow enough that hallucinations still fail.
        assert normalize("entirely different") not in normalize(CHUNK_TEXT)


class TestChunking:
    def test_chunk_ids_are_stable_and_positional(self):
        assert build_chunk_id(PAPER, 4, 2) == f"{PAPER}:p4:c2"

    def test_pages_are_preserved_on_every_chunk(self):
        pages = [ParsedPage(page_number=n, text="sentence. " * 200) for n in (1, 2)]
        documents = chunk_pages(PAPER, pages)
        assert {d.metadata["page_number"] for d in documents} == {1, 2}
        assert all(d.metadata["arxiv_id"] == PAPER for d in documents)

    def test_long_pages_split_into_several_chunks(self):
        pages = [ParsedPage(page_number=1, text="sentence. " * 500)]
        assert len(chunk_pages(PAPER, pages)) > 1

    def test_blank_pages_are_skipped(self):
        pages = [
            ParsedPage(page_number=1, text="   \n  "),
            ParsedPage(page_number=2, text="real content here"),
        ]
        documents = chunk_pages(PAPER, pages)
        assert [d.metadata["page_number"] for d in documents] == [2]

    def test_chunk_ids_are_unique(self):
        pages = [ParsedPage(page_number=n, text="sentence. " * 300) for n in (1, 2, 3)]
        ids = [d.metadata["chunk_id"] for d in chunk_pages(PAPER, pages)]
        assert len(ids) == len(set(ids))


class TestSupportJudge:
    def test_unsupported_citation_is_dropped(self, monkeypatch):
        monkeypatch.setattr(analysis, "judge_support", grading(0))
        kept, dropped_claims, unsupported = apply_support_judge(
            [supported_claim("over-sample easy queries")]
        )
        assert kept == [] and dropped_claims == 1 and unsupported == 1

    def test_partial_support_survives(self, monkeypatch):
        # 9 of the 40 hand-labeled citations are partials, all of them enumerations
        # where the claim lists more than the quote names. Dropping those would
        # discard mostly-correct work, so the threshold keeps them.
        monkeypatch.setattr(analysis, "judge_support", grading(1))
        kept, dropped_claims, unsupported = apply_support_judge(
            [supported_claim("over-sample easy queries")]
        )
        assert len(kept) == 1 and dropped_claims == 0 and unsupported == 0
        assert SUPPORT_THRESHOLD == 1

    def test_grade_is_recorded_on_the_surviving_citation(self, monkeypatch):
        monkeypatch.setattr(analysis, "judge_support", grading(2))
        kept, _dropped, _unsupported = apply_support_judge(
            [supported_claim("over-sample easy queries")]
        )
        assert kept[0].evidence[0].support_grade == 2

    def test_claim_keeps_its_supported_citation_and_drops_the_other(self, monkeypatch):
        monkeypatch.setattr(analysis, "judge_support", grading(2, 0))
        kept, dropped_claims, unsupported = apply_support_judge(
            [supported_claim("the supported quote", "the unsupported quote")]
        )
        assert len(kept) == 1 and len(kept[0].evidence) == 1
        assert kept[0].evidence[0].excerpt == "the supported quote"
        assert dropped_claims == 0 and unsupported == 1

    def test_a_missing_verdict_fails_closed(self, monkeypatch):
        # An unjudged citation must not reach the report because the reply had a hole
        # in it. Silence is not support.
        monkeypatch.setattr(analysis, "judge_support", grading(None))
        kept, dropped_claims, unsupported = apply_support_judge(
            [supported_claim("over-sample easy queries")]
        )
        assert kept == [] and dropped_claims == 1 and unsupported == 1

    def test_grades_are_consumed_in_citation_order_across_claims(self, monkeypatch):
        monkeypatch.setattr(analysis, "judge_support", grading(0, 2))
        kept, _dropped, _unsupported = apply_support_judge(
            [
                supported_claim("first quote", text="first claim"),
                supported_claim("second quote", text="second claim"),
            ]
        )
        assert [claim.text for claim in kept] == ["second claim"]

    def test_no_claims_means_no_model_call(self, monkeypatch):
        def explode(items):
            raise AssertionError("the judge must not be called with nothing to grade")

        monkeypatch.setattr(analysis, "judge_support", explode)
        assert apply_support_judge([]) == ([], 0, 0)


class TestJudgeReplyHandling:
    """`judge_support` reads one batched reply, so it has to survive a ragged one."""

    def reply(self, monkeypatch, *verdicts: SupportVerdict) -> None:
        graded = SupportVerdicts(verdicts=list(verdicts))
        monkeypatch.setattr(
            analysis, "generate_structured", lambda prompt, result_type: graded
        )

    def test_verdicts_are_matched_by_index_not_reply_order(self, monkeypatch):
        self.reply(
            monkeypatch,
            SupportVerdict(index=2, grade=0, reason="wrong sentence"),
            SupportVerdict(index=1, grade=2, reason="states the claim"),
        )
        assert judge_support([("claim one", "quote one"), ("claim two", "quote two")]) == [2, 0]

    def test_an_index_that_was_never_sent_is_ignored(self, monkeypatch):
        self.reply(monkeypatch, SupportVerdict(index=7, grade=2, reason="invented"))
        assert judge_support([("claim", "quote")]) == [None]

    def test_an_omitted_item_comes_back_unjudged(self, monkeypatch):
        self.reply(monkeypatch, SupportVerdict(index=1, grade=2, reason="only one"))
        assert judge_support([("claim one", "quote one"), ("claim two", "quote two")]) == [2, None]

    def test_every_item_is_numbered_in_the_prompt(self, monkeypatch):
        seen = {}

        def capture(prompt, result_type):
            seen["prompt"] = prompt
            return SupportVerdicts(verdicts=[])

        monkeypatch.setattr(analysis, "generate_structured", capture)
        judge_support([("claim one", "quote one"), ("claim two", "quote two")])

        assert "1.\nClaim: claim one\nExcerpt: quote one" in seen["prompt"]
        assert "2.\nClaim: claim two\nExcerpt: quote two" in seen["prompt"]
