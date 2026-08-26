# Verification of the example review

`example_review.md` was produced by the `judged-example-2` run and carries **76
citations** across 4 papers. This records what was checked, by what means, and what a
reader found that no check could.

## How it was produced

```bash
python arxiv_lit_reviewer.py run \
  --query "How is mechanistic interpretability used to understand transformer internals?" \
  --thread-id judged-example-2 \
  --output examples/example_review.md
```

Run on 2026-08-26 at the shipping defaults: `--max-results 30`, `--target-papers 4`,
`--retriever hybrid-rerank`, `--top-k 5`, relevance threshold 3. Thirty candidates were
screened from abstracts, four papers were selected and downloaded, 261 chunks were
indexed, and 76 of the 80 citations the model proposed survived validation.

## Checked automatically

These run as code and are reproducible; nothing here relies on judgment.

| Check | Result |
| --- | --- |
| Every cited `chunk_id` exists in the run's index | 76/76 |
| Every cited chunk belongs to the paper it is attributed to | 76/76 |
| Every excerpt occurs verbatim in the chunk it cites | 76/76 |
| Every page anchor matches the cited chunk's page | 76/76 |
| Every page link in the report traces to a validated citation | 24/24 |
| Papers cited in the report were actually analyzed | 4/4 |

The fifth row is the one worth noting: synthesis is a language model writing Markdown,
so it could in principle invent a page link. None were invented.

## Read by hand

All 76 citations were read against the pages they cite and graded **OK** (the excerpt
establishes the claim), **WEAK** (it supports part of it), or **NO** (it does not support
it). The filled sheet is `verification_sheet.md`, kept so the grades can be audited
rather than taken on trust.

| Verdict | Count |
| --- | --- |
| OK | 63 |
| WEAK | 13 |
| NO | 0 |

Nothing was rejected. Every excerpt said something the claim was built on; in the 13 weak
cases the question is how much.

## Reader against the support judge

The support judge graded these same 76 citations during the run, so the two can be
compared directly.

| | reader OK | reader WEAK |
| --- | --- | --- |
| judge `2` establishes | 62 | 6 |
| judge `1` supports partly | 1 | 7 |

Exact agreement is **90.8%** (69 of 76). The seven disagreements run in both directions:
six where the judge was more generous than the reader, one where it was stricter. Every
one is about scope rather than about the quote.

- **#33** is the widest. The claim says the paper "addresses the lack of frameworks to
  study model representations, as current methods fail to yield verifiable
  interpretations". The excerpt reads, in full, "We posit the need of new frameworks to
  think about and study representations." The need is quoted; the diagnosis attached to
  it is not.
- **#10, #19, #27, #34, #43** are the same shape, milder: the claim aggregates or
  generalises slightly past the sentence quoted for it.
- **#29** runs the other way. The claim says the authors use Marks & Tegmark's method to
  compute linear directions for feature representations; the excerpt says exactly that,
  scoped to truthfulness in one model. The judge called it partial, the reader called it
  established.

## What this does and does not establish

It measures agreement, not correctness, and it measures it on kept citations only. The
judge had already discarded one citation before this sheet existed, so nothing here
scores it on what it rejects — `evals/run_claim_judge.py` does that, against a set built
to contain failures.

No `NO` verdict appeared anywhere in the 76, which matches the 40-citation sample in
[`DESIGN.md`](../DESIGN.md#4-grounding) and means a catch rate cannot be estimated from
this run either.

The claims are the model's, drawn only from retrieved excerpts of the cited papers.
Verification shows an excerpt exists where it is claimed to and that a reader agrees it
supports the claim built on it. It says nothing about whether the paper is right.
