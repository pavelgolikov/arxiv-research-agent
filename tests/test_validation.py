"""Citation validation and chunking.

`validate_claim` is the anti-hallucination guarantee: a claim survives only if it
cites a chunk that was actually shown, belonging to the paper under analysis, quoting
text that genuinely occurs there. These tests pin both directions — faithful citations
must survive, and every way of being unfaithful must not.
"""

import pytest
from langchain_core.documents import Document

from arxiv_reviewer.analysis import (
    EVIDENCE_EXCERPT_CHARS,
    normalize,
    validate_claim,
)
from arxiv_reviewer.rag import build_chunk_id, chunk_pages
from arxiv_reviewer.review_types import DraftClaim, DraftEvidence, ParsedPage

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
