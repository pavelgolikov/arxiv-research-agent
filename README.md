# arXiv Research Agent

A LangGraph workflow that searches arXiv, screens candidates against a research
question, indexes the selected papers into a vector store, extracts claims that cite
the pages they came from, and writes a Markdown literature review with Gemini through
LangChain.

**Built with:** LangGraph, LangChain, Chroma, BM25, cross-encoder reranking, Gemini,
Pydantic, SQLite, PyMuPDF, pytest.

Example output: [`examples/example_review.md`](examples/example_review.md).
Its citations were machine-checked and manually verified —
[`examples/VERIFICATION.md`](examples/VERIFICATION.md).

## Features

**Retrieval and RAG** — page-preserving PDF parsing; chunking with stable `chunk_id`s;
persistent Chroma vector index; dense, BM25, hybrid reciprocal-rank-fusion, and
cross-encoder reranking retrievers; multi-query expansion.

**Agent orchestration (LangGraph)** — `Send` map-reduce fan-out for screening and
analysis; reducer-backed state with deterministic ordering independent of concurrency;
SQLite checkpointing behind `run` / `resume` / `status`; typed per-branch failures with
retry classification and partial reports.

**Grounding and citation validation** — per-facet retrieval scoped to one paper; claims
that must carry chunk-level evidence; three deterministic checks that the quote exists
where it says it does, then a support judge that discards quotes which do not support the
claim; page-anchored citations into the source PDF.

**Evaluation and benchmarking** — two hand-labeled datasets (50 retrieval questions with
312 judged chunks; 7 screening queries × 12 candidates); TREC-style pooling with measured
pooling bias; four-way retrieval ablation with paired-bootstrap confidence intervals;
threshold sweep over the real selection rule; groundedness metrics with independent
re-validation; a hand-labeled claim-support sample, extended with constructed failures so
the support judge can be scored against it; reproducible index rebuild guarded by a
pool-coverage check.

**Engineering** — 151 tests requiring no network access and no API key; published
numbers generated from committed JSON; graceful exit codes; a worked example with its
verification record.

Not included: LangSmith tracing, human-in-the-loop `interrupt`, `pyproject.toml`
packaging, CI, web UI.

## Repository layout

- `arxiv_lit_reviewer.py` — command-line entry point (`run`, `resume`, `status`).
- `arxiv_reviewer/` — application package:
  - `workflow.py` — graph construction, fan-out, SQLite persistence, run/resume/status.
  - `retrieval.py` — arXiv search, PDF download, page-preserving parsing.
  - `rag.py` — chunking, embedding, Chroma index, retriever strategies.
  - `analysis.py` — candidate screening and grounded per-facet analysis.
  - `reporting.py` — Markdown synthesis, deterministic fallback rendering.
  - `review_types.py` — Pydantic models and the typed graph state.
  - `failures.py` — retry classification and retrying.
  - `gemini_client.py` — model access through LangChain.
- `evals/`
  - `build/` — dataset construction: arXiv search, corpus parsing, question generation,
    candidate pooling, offline labeling page.
  - `metrics.py` — MRR, nDCG, recall, paired bootstrap.
  - `run_retrieval.py`, `run_screening.py`, `run_groundedness.py`,
    `run_claim_judge.py` — metric runners.
  - `measure_categories.py`, `measure_depth.py` — search-quality studies.
  - `build/claim_support.py` — samples citations for hand-grading, builds the judge
    evaluation set, and collects both into `labels/`.
  - `build_index.py` — rebuilds the vector index from committed chunks; `--verify`
    checks that the labels cover everything the retrievers return.
  - `render_tables.py` — regenerates the tables in `README.md` and `DESIGN.md`.
  - `data/`, `labels/`, `results/` — frozen datasets, hand labels, metric output.
    Committed. `index/` is not.
- `examples/` — one complete run with its verification record.
- `tests/` — pytest suite.
- `DESIGN.md` — evaluation methodology, experiments, rejected alternatives.

Run state lives under `.arxiv-reviewer/` (git-ignored): `checkpoints.sqlite` and
`chroma/`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in a `.env` file at the repository root.

The cross-encoder reranker requires `sentence-transformers` and `torch`. Other
`--retriever` values do not load the model at runtime.

## Run

```bash
.venv/bin/python arxiv_lit_reviewer.py run \
  --query "What is the latest and greatest in model self-improvement?"
```

The command prints a thread ID before starting work:

```bash
.venv/bin/python arxiv_lit_reviewer.py status --thread-id THREAD_ID
.venv/bin/python arxiv_lit_reviewer.py resume --thread-id THREAD_ID
```

`status` reads the checkpoint database only: no API key, no network calls. `resume`
restarts at the last incomplete step; finished branches are not re-executed.

Exit codes: `0` when a report was produced, including `partial` and `empty` runs; `1`
when the run could not continue or was interrupted; `2` for invalid arguments or an
unknown thread ID. A failed run prints the error and the thread ID, and the thread
stays resumable.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--query` | required | Research question to review. |
| `--thread-id` | generated UUID | Name of the run. |
| `--max-results` | 30 | Candidate papers to retrieve from arXiv, shared across the planned queries. |
| `--target-papers` | 4 | Relevant papers to include in the review. |
| `--retriever` | `hybrid-rerank` | Retrieval strategy over indexed chunks. |
| `--top-k` | 5 | Chunks returned per question. |
| `--fetch-k` | 20 | Candidates fused before reranking. |
| `--multi-query` | off | Expand each question into paraphrases first. |
| `--max-concurrency` | 3 | Branches processed in parallel. |
| `--data-dir` | `.arxiv-reviewer` | Checkpoint and vector-store location. |
| `--output` | `review.md` | Report path. |

## Pipeline

```mermaid
flowchart TD
    START([start]) --> plan[plan search<br/>1-3 arXiv queries]
    plan --> search[search arXiv<br/>rate-limited, deduplicated]
    search -->|no candidates| render
    search --> screen{{"screen each candidate<br/>Send fan-out, abstract only"}}
    screen --> select[select papers<br/>score desc, then search position]
    select -->|none selected| render
    select --> analyze{{"analyze each paper<br/>Send fan-out: download, chunk,<br/>embed, retrieve per facet"}}
    analyze --> validate[validate every citation<br/>3 deterministic checks,<br/>then a support judge]
    validate --> synth[synthesize]
    synth -->|synthesis fails| render
    synth --> render[render Markdown<br/>written atomically]
    render --> DONE([end])
```

Screening reads title, authors, date, and abstract only. PDFs are downloaded for
selected papers.

Screening and analysis fan out with LangGraph `Send` into reducer-backed lists, then
sort by (score descending, original search position). Output is identical at any
`--max-concurrency`.

A failing branch does not stop its siblings. Timeouts, connection errors, HTTP
408/425/429/5xx, and provider `RESOURCE_EXHAUSTED` / `UNAVAILABLE` /
`DEADLINE_EXCEEDED` / `INTERNAL` are retried three times with exponential backoff and
jitter. Other errors — unparseable PDF, oversized download, schema violation — are
recorded as typed failures. The run finishes with whatever succeeded, the report is
marked `partial`, and failures are listed in a `Failures` section.

Retries run inside each branch. `RetryPolicy` is applied to `search` only.

Interrupted runs resume from the last checkpoint.

## Retrieval

Paper text is split into overlapping ~1000-character chunks carrying their page numbers,
embedded with `models/gemini-embedding-001`, and stored in a persistent Chroma index.

| `--retriever` | Strategy |
| --- | --- |
| `dense` | Embedding similarity only. |
| `bm25` | Keyword frequency only. |
| `hybrid` | Both, fused with reciprocal rank fusion. |
| `hybrid-rerank` | Hybrid candidates rescored by a `ms-marco-MiniLM-L-6-v2` cross-encoder. |

`--multi-query` expands each question into paraphrases and retrieves for all of them.
With reranking enabled the expanded candidates are fused and rescored once, honouring
`--top-k`.

## Grounding

Each selected paper is analyzed one facet at a time — research problem, method,
experimental setup, main findings, limitations, relevance to the query — from chunks
retrieved for that facet and scoped to that paper.

Every statement is returned as a claim carrying at least one citation: a `chunk_id` plus
an excerpt quoted from that chunk. Three checks run first, without a model call:

1. the cited chunk exists among the chunks the model was shown,
2. it belongs to the paper being analyzed, and
3. the quoted excerpt occurs in that chunk, ignoring case, whitespace, and hyphenation.

Those three prove the quote is real. None of them can tell whether the quote actually
backs up the sentence it was attached to, so a fourth check asks exactly that. A model
reads each claim next to its quote and grades the pair:

- `2` — the quote establishes the claim,
- `1` — the quote supports part of it,
- `0` — the quote does not support it.

Anything graded `0` is thrown out, and so is any citation the model skipped instead of
grading. That costs one model call per facet, not one per citation.

Grade `1` is kept. The partial grades measured so far are all one shape: the claim lists
five things and the quote names two of them, so it supports part of what was written.
Throwing those out would delete correct work to fix a sentence-phrasing problem.

Discarded citations take nothing with them except a claim left with no citations at all,
which is discarded too. The report records both counts. Surviving claims render as
page-anchored links into the source PDF, carrying the grade they were given.

## Evaluation

Two hand-labeled datasets under `evals/`, frozen and committed. Tables below are
generated from `evals/results/*.json` by `python -m evals.render_tables --write`.

| Dataset | Size |
| --- | --- |
| Retrieval | 5 papers, 570 chunks, 50 questions, 312 judged-relevant chunks (246 fully answering, 66 partial) |
| Screening | 7 queries x 12 candidates, each labeled irrelevant / related / central |

Thirty of the fifty retrieval questions are the strings `analysis.py` asks of every
paper. The other twenty are paper-specific factual questions.

Methodology, experiments, and rejected alternatives: [`DESIGN.md`](DESIGN.md).

### Retrieval ablation

Four strategies, fifty questions, scored at pool depth.

<!-- eval:retrieval -->
| Strategy | MRR | nDCG@10 | recall@5 (specific) | recall@5 (facet) |
| --- | --- | --- | --- | --- |
| `dense` | 0.736 | 0.586 | 0.541 | 0.299 |
| `bm25` | 0.627 | 0.423 | 0.345 | 0.264 |
| `hybrid` | 0.750 | 0.606 | 0.443 | 0.371 |
| `hybrid-rerank` | 0.771 | 0.623 | 0.528 | 0.391 |
| _best achievable_ | — | — | _0.921_ | _0.714_ |
<!-- /eval:retrieval -->

Paired-bootstrap 95% intervals exclude zero for 5 of 16 pairwise comparisons; full
intervals in [`DESIGN.md`](DESIGN.md#2-reading-the-ablation-honestly). Recall is split
by question type and reported against the best score achievable at that cutoff.
`--multi-query` is not scored: its paraphrases retrieve chunks outside the frozen pools.

### Screening quality

Threshold sweep over `select_papers_node`, scored against the labels.

<!-- eval:screening -->
| Threshold | precision@4 (central) | precision@4 (related) | central recall | best achievable | queries under-filled |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.786 | 0.964 | 0.683 | _0.730_ | 0 |
| 2 | 0.810 | 1.000 | 0.683 | _0.730_ | 1 |
| 3 (current, **recommended**) | 0.810 | 1.000 | 0.683 | _0.730_ | 1 |
| 4 | 0.810 | 1.000 | 0.611 | _0.730_ | 2 |
| 5 | 0.786 | 0.857 | 0.540 | _0.730_ | 2 |

| Model score | labeled irrelevant | labeled related | labeled central |
| --- | --- | --- | --- |
| 1 | 24 | 0 | 0 |
| 2 | 4 | 1 | 0 |
| 3 | 0 | 2 | 1 |
| 4 | 0 | 5 | 4 |
| 5 | 0 | 4 | 39 |
<!-- /eval:screening -->

Threshold 3 is the shipped default. Per-query figures are in
`evals/results/screening.json`. Two search-quality studies —
[`evals/results/categories.json`](evals/results/categories.json) and
[`evals/results/search_depth.json`](evals/results/search_depth.json) — set
`--max-results` to 30 and rejected an arXiv category filter.

### Groundedness

Two live runs at the shipping defaults, read from the LangGraph checkpoints.

<!-- eval:groundedness -->
| Measure | Value |
| --- | --- |
| Papers analyzed | 8 across 2 runs |
| Claim-support rate | **94.4%** (152 of 161 proposed claims kept) |
| Citation referential integrity | **94.3%** (165 of 175 proposed citations resolved) |
| Citation support integrity | not judged — run predates the check |
| Citations per surviving claim | 1.09 |
| Independent re-validation | 100.0% (165 citations re-checked, 0 failures) |
<!-- /eval:groundedness -->

Rates count citations and claims surviving validation out of those the model proposed.

### Claim support

Validation runs in two layers: three exact checks prove the quote exists where it says
it does, then a model grades whether that quote actually supports the claim, and the
failures are discarded.

How well that model grades is measured against citations a person graded by hand — 40 of
the 165 above, picked at random and read one by one against their claims. They were
graded before the support check existed, so nothing the judge does could have shaped
them.

<!-- eval:claim_support -->
| Measure | Rate | 95% CI |
| --- | --- | --- |
| Excerpt establishes the claim | **77.5%** (31 of 40) | [62%, 88%] |
| Excerpt supports it at least partly | **100.0%** (31 + 9 of 40) | [91%, 100%] |
| Excerpt does not support the claim | 0 of 40 | — |
<!-- /eval:claim_support -->

Partial grades concentrate in one facet:

<!-- eval:claim_support_facets -->
| Facet | Partial grades |
| --- | --- |
| `experimental_setup` | 6 of 8 |
| `limitations` | 0 of 7 |
| `main_findings` | 2 of 9 |
| `method` | 0 of 5 |
| `relevance_to_query` | 0 of 4 |
| `research_problem` | 1 of 7 |
<!-- /eval:claim_support_facets -->

### Scoring the support judge

Those 40 cannot score the judge on their own, because **not one of them is a `0`**.
Against a set containing no failures, a model that calls everything "supported" is right
every single time. To find out whether the judge rejects bad citations, the set has to
contain some.

So 30 more were graded: 10 further real citations, and 20 where the quote was replaced
with a different quote from the same paper. That swap is the failure the three
deterministic checks cannot see — the quote is real, it is in the right paper, it is on
the page it names. It just belongs to a different sentence.

The 20 swaps sit shuffled among the 10 real ones with nothing marking which is which,
and they are read and graded like everything else rather than written down as `0` by
assumption. That assumption would have been wrong on **5 of the 20**: a quote pulled
from elsewhere in the same paper still supported part of the claim it landed on. Scoring
those as failures would have marked the judge wrong for keeping them.

<!-- eval:claim_judge -->
| Measure | Rate | 95% CI |
| --- | --- | --- |
| Rejects a citation a reader also rejected (catch rate) | **100.0%** (15 of 15) | [80%, 100%] |
| Rejects a citation a reader kept (false-drop rate) | **3.6%** (2 of 55) | [1%, 12%] |
| Exact grade agreement | **87.1%** (61 of 70) | [77%, 93%] |
<!-- /eval:claim_judge -->

Two numbers come out of this, and they measure opposite mistakes:

- **catch rate** — of the citations a person rejected, how many the judge also rejected.
  This is what the check is worth.
- **false-drop rate** — of the citations a person accepted, how many the judge threw out.
  This is what it costs, in correct work deleted from the report.

A judge that rejects everything scores a perfect catch rate, so neither number means
anything without the other.

Both false drops landed on swapped citations that turned out to support their new claim
anyway — two of those five ambiguous items. Against the 50 citations the pipeline
actually produced, it dropped none: **0 of 50**. The pooled 3.6% is the figure to quote,
since those five are real disagreements about genuinely borderline quotes, but the two
error types are not spread evenly and the real citations are what a report is built from.

The 77.5% and 100% figures further up come from the random 40 alone, so the 20
deliberately broken citations cannot drag them down.

`evals/labels/claim_support_labels.json` carries each grade with its claim and excerpt.
Regenerate the sheets with `python -m evals.build.claim_support` and
`python -m evals.build.claim_support --judge-set`, then score the judge with
`python -m evals.run_claim_judge`.

## Tests

```bash
.venv/bin/python -m pytest
```

151 tests, no network access and no API key. An autouse fixture fails outbound
connections. Chroma runs against a temporary directory with deterministic embeddings.

Coverage: graph terminal paths (no candidates, none selected, success, partial branch
failure, synthesis fallback); concurrency 1 versus 3 producing identical output;
citation validation against hallucinated chunk IDs, wrong-paper chunks, paraphrases,
and PDF hyphenation; the support judge dropping unsupported citations, keeping partial
ones, and failing closed on a verdict it never received; SQLite round trip and resume;
retry classification and backoff; the eval pool-coverage guard and the judge's scoring
arithmetic.

## Limitations

- arXiv is the only source.
- Candidate quality is bounded by arXiv search. A category filter was measured and
  rejected; see [`DESIGN.md`](DESIGN.md#3-screening-and-search-quality).
- `--max-results` is divided across the planned queries.
- The retrieval ablation has 50 questions.
- PDF text extraction quality varies, especially for tables, figures, and formulae.
- The support check is a model grading another model's work, so it is only as good as
  its measured agreement with a person. The failures it was tested against are quotes
  swapped between claims — the obvious kind. Nothing here shows it catches a quote that
  supports most of a claim but not the qualifier attached to it.
- arXiv results are not peer reviewed.
- Runs cost one model call per candidate screened (30 by default) plus twelve per
  selected paper: six to analyze its facets, six to judge the citations they produced.
