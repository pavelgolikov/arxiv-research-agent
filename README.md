# arXiv Research Agent

A LangGraph workflow that searches arXiv, screens candidates against a research
question, indexes the selected papers into a vector store, extracts claims that
cite the pages they came from, and writes a Markdown literature review with Gemini
through LangChain.

## Project status

The pipeline runs end to end. The hand-labeled datasets the evaluation rests on are
frozen and committed; the ablation and groundedness runs that consume them are in
progress. No performance numbers appear here until those runs are committed.

**Working today**

| Area | State |
| --- | --- |
| Page-preserving PDF parsing, chunking with stable `chunk_id`s | done |
| Chroma vector index, persisted per run thread | done |
| Dense / BM25 / hybrid (RRF) / cross-encoder rerank retrieval | done |
| Multi-query expansion | done |
| Per-facet grounded analysis with deterministic citation validation | done |
| Page-anchored citations in the report | done |
| Parallel screening and analysis via LangGraph `Send` + reducers | done |
| Deterministic ordering independent of concurrency | done |
| Abstract-first screening (PDFs fetched only for selected papers) | done |
| Retries, typed per-branch failures, partial reports | done |
| SQLite checkpointing with real `run` / `resume` / `status` | done |
| Labeled retrieval dataset — 50 questions, 312 judged chunks | done |
| Labeled screening dataset — 7 queries x 12 candidates | done |
| Reproducible index rebuild with pool-coverage verification | done |

**Not built yet**

| Area | Notes |
| --- | --- |
| Retrieval ablation | recall@5, MRR, nDCG@10 across the four `--retriever` settings |
| Groundedness metrics | citation referential integrity and claim-support rate over generated reports |
| `evals/results/*.json` | machine-readable output; the only place performance numbers may come from |
| Automated test suite | no `tests/` directory exists; see Verification below |
| Verified example report | one report with every citation manually checked, committed under `examples/` |
| Repository cleanup | `results/` still holds pre-rewrite artifacts |
| Graceful top-level failure | `EXIT_FAILED` is defined but never returned; a mid-run crash prints a traceback |

Deliberately out of scope for now: LangSmith tracing, human-in-the-loop
`interrupt`, `pyproject.toml` packaging, CI, and any web UI or service.

See `PORTFOLIO_PLAN.md` for the full plan and the reasoning behind it.

## Repository layout

- `arxiv_lit_reviewer.py` — command-line entry point (`run`, `resume`, `status`).
- `arxiv_reviewer/` — application package, split by responsibility:
  - `workflow.py` — graph construction, fan-out, SQLite persistence, run/resume/status.
  - `retrieval.py` — arXiv search, PDF download, page-preserving parsing.
  - `rag.py` — chunking, embedding, Chroma index, retriever strategies.
  - `analysis.py` — candidate screening and grounded per-facet analysis.
  - `reporting.py` — Markdown synthesis, deterministic fallback rendering.
  - `review_types.py` — Pydantic models and the typed graph state.
  - `failures.py` — retry classification and retrying.
  - `gemini_client.py` — model access through LangChain.
- `evals/` — evaluation, split by direction of data flow:
  - `build/` — dataset construction: arXiv search, corpus parsing, question
    generation, candidate pooling, and the offline labeling page. Run rarely.
  - `build_index.py` — rebuilds the vector index from the committed chunks and
    verifies that the labels still cover everything the retrievers return.
  - `data/`, `labels/`, `results/` — the frozen datasets, the hand labels, and the
    metric output. All committed; `index/` is not.
- `results/` — artifacts from the **pre-rewrite prototype**, kept only for reference.
  Its checkpoint JSON files use a state schema this code no longer has, and nothing
  reads them. Slated for removal.
- `PORTFOLIO_PLAN.md` — roadmap for the current upgrade.

Run state lives under `.arxiv-reviewer/` (git-ignored): `checkpoints.sqlite` holds
LangGraph checkpoints and `chroma/` holds the persistent vector index.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in a `.env` file at the repository root.

The cross-encoder reranker pulls in `sentence-transformers` and `torch`, which
dominate install size. Selecting a different `--retriever` avoids loading the
model at runtime, but the dependency is still installed.

## Run

```bash
.venv/bin/python arxiv_lit_reviewer.py run \
  --query "What is the latest and greatest in model self-improvement?"
```

The command prints a thread ID before it starts work. Use it to inspect or continue
the run:

```bash
.venv/bin/python arxiv_lit_reviewer.py status --thread-id THREAD_ID
.venv/bin/python arxiv_lit_reviewer.py resume --thread-id THREAD_ID
```

`status` reads the checkpoint database only, so it needs no API key and makes no
network calls. `resume` restarts at the last incomplete step, so papers that were
already downloaded, indexed, or analyzed are not processed again.

Exit codes: `0` when a report was produced, `2` for invalid arguments or an
unknown thread ID. A run that fails part-way currently surfaces as an unhandled
traceback rather than a deliberate exit code; the thread is still resumable.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--query` | required | Research question to review. |
| `--thread-id` | generated UUID | Name of the run. |
| `--max-results` | 10 | Candidate papers to retrieve from arXiv. |
| `--target-papers` | 4 | Relevant papers to include in the review. |
| `--retriever` | `hybrid-rerank` | Retrieval strategy over indexed chunks. |
| `--top-k` | 5 | Chunks returned per question. |
| `--fetch-k` | 20 | Candidates fused before reranking. |
| `--multi-query` | off | Expand each question into paraphrases first. |
| `--max-concurrency` | 3 | Branches processed in parallel. |
| `--data-dir` | `.arxiv-reviewer` | Checkpoint and vector-store location. |
| `--output` | `review.md` | Report path. |

## Pipeline

```text
search  ->  screen every candidate in parallel (abstract only, no PDF)
        ->  rank by score then original search position, select target-papers
        ->  analyze each selected paper in parallel (download, index, cite)
        ->  synthesize the report
```

Screening reads only title, authors, date, and abstract, so PDFs are downloaded
only for papers that survive selection. Both stages fan out with LangGraph `Send`
and collect into reducer-backed lists, which arrive in completion order and are
then sorted deterministically: identical reports come out at any
`--max-concurrency`.

A branch that fails does not stop its siblings. Transient failures (timeouts,
connection errors, HTTP 408/425/429/5xx, provider `RESOURCE_EXHAUSTED` /
`UNAVAILABLE` / `DEADLINE_EXCEEDED` / `INTERNAL`) are retried three times with
exponential backoff and jitter. Anything else — an unparseable PDF, an
oversized download, a schema violation — is recorded as a typed failure. The run
finishes with whatever succeeded, the report is marked `partial`, and every
failure is listed in a `Failures` section appended after synthesis so it cannot
be omitted.

Retrying happens inside each branch rather than through LangGraph's node-level
`RetryPolicy`. In langgraph 1.2.10 a node `error_handler` runs but does not
suppress the original exception for tasks dispatched by `Send`, so a single bad
paper would still abort the whole run. `RetryPolicy` is kept on `search`, where
failing the run and resuming later is the intended behavior.

Interrupted runs resume from the last checkpoint. Branches that already finished
are not re-executed, so their downloads and model calls are not paid for twice.

## Retrieval

Paper text is split into overlapping ~1000-character chunks that keep their page
numbers, embedded with `models/gemini-embedding-001`, and stored in a persistent
Chroma index under `.arxiv-reviewer/chroma/`.

| `--retriever` | Strategy |
| --- | --- |
| `dense` | Embedding similarity only. |
| `bm25` | Keyword frequency only. |
| `hybrid` | Both, fused with reciprocal rank fusion. |
| `hybrid-rerank` | Hybrid candidates rescored by a `ms-marco-MiniLM-L-6-v2` cross-encoder. |

`--multi-query` expands each question into paraphrases and retrieves for all of
them. When reranking is enabled the expanded candidates are fused and rescored
once, so the result still honours `--top-k`.

Which of these is actually better on this corpus is an open question until the
ablation in `evals/` exists. No comparative claim is made here.

## Grounding

Papers are not summarized from their full text. Each selected paper is analyzed
one facet at a time — research problem, method, experimental setup, main
findings, limitations, and relevance to the query — and each facet is answered
only from chunks retrieved for that facet, scoped to that paper.

The model must return every statement as a claim carrying at least one citation,
where a citation is a `chunk_id` plus an excerpt quoted from that chunk. Each
citation is then checked without calling a model:

1. the cited chunk must exist among the chunks the model was shown,
2. it must belong to the paper being analyzed, and
3. the quoted excerpt must actually occur in that chunk, ignoring whitespace and case.

Citations failing any check are discarded, and a claim left with no surviving
citation is discarded with it. The report records how many were dropped, so a
paper whose analysis was partly rejected is visible rather than silently thinner.
Surviving claims render as page-anchored links into the source PDF.

## Evaluation

Two hand-labeled datasets live under `evals/`, frozen and committed so results are
reproducible and do not drift as arXiv re-ranks its search results.

| Dataset | Size |
| --- | --- |
| Retrieval | 5 papers, 570 chunks, 50 questions, 312 judged-relevant chunks (246 fully answering, 66 partial) |
| Screening | 7 queries x 12 candidates, each labeled irrelevant / related / central |

Thirty of the fifty retrieval questions are the exact strings `analysis.py` asks of
every paper, so the benchmark measures the real workload rather than a proxy. The
other twenty are paper-specific factual questions — named datasets, baselines,
numbers — that probe precise retrieval.

### Pooling, and measuring what it misses

Judging every chunk against every question would be 5,700 decisions. Instead the
dataset uses **pooling**, the standard TREC approach: each of the four retrievers
contributes its top 10 for a question, and only the union is judged — 14 to 23
chunks per question instead of the whole paper.

Pooling has a known flaw. A chunk that no retriever returns is never judged, so it
counts as irrelevant by default and recall comes out higher than it should be.
Rather than assume the pools were deep enough, three chunks per question that **no**
retriever returned were sampled into the pool and judged blind alongside the rest.
If those come back relevant, the pools were too shallow.

They came back relevant more often than the 5% budgeted for, and how they split is
the useful part:

| Question type | Sampled unpooled chunks judged relevant |
| --- | --- |
| Facet — "what are the main findings?" | 14 / 90 = **15.6%** |
| Paper-specific — "which datasets were used?" | 0 / 60 = **0%** |
| Overall | 14 / 150 = 9.3% |

This is not a pool-depth problem, and deepening the pools would not fix it. Those
chunks were sampled from chunks *no retriever ranked at all*, whereas raising the
depth to 20 reaches ranks 11-20 — a different population. The actual cause is that
broad facet questions have diffuse relevance: "what are the main quantitative
results?" is genuinely answered by dozens of chunks spread through a 157-chunk
paper, while a precise question has two or three answers and the retrievers find
them. So the pools were kept as they are and the metrics were chosen to suit them.

### What the bias can and cannot touch

Pool depth is 10 and every retriever contributed its top 10, so at any cutoff
k <= 10 **every chunk appearing in a ranked list has a label**. No unjudged chunk can
enter a result list and be silently scored as irrelevant. The numerator of every
metric is therefore exact, and only denominators are exposed:

| Metric | Exposure |
| --- | --- |
| MRR | None. Every ranked chunk is judged. |
| nDCG@10 | Ideal DCG only, on the 24 of 30 facet questions whose ideal top 10 is not already filled with top-grade chunks — and identically for all four retrievers, since they share one ground truth. The absolute level is slightly optimistic; the comparison between retrievers is not. |
| recall@5 | Directly. The size of the relevant set is exactly what the missed chunks corrupt. |

recall@5 was the weakest metric here even before this: with a mean of 7.7 relevant
chunks per facet question, five slots cannot hold them all, so it is capped at 71%
on facet questions however good the retriever is — 92% on paper-specific ones,
where the relevant sets are smaller.

The ablation therefore leads with **MRR and nDCG@10**, and reports **recall@5 split
by question type** rather than pooled into a single figure: clean on the
paper-specific half, flagged on the facet half. `bpref` and `infAP`, estimators
built for incomplete judgments, are the fully rigorous alternative and are not
implemented here.

### Keeping the guarantee true

"Every ranked chunk is judged" holds only while the index matches the one the pools
were built from. `evals/index/` is not committed, so `python -m evals.build_index`
rebuilds it from the committed chunk file — identical text, no PDF downloads — and
`--verify` replays all four retrievers over all fifty questions to confirm every
retrieved chunk was judged, writing `evals/results/index_coverage.json`.

Re-embedding the entire corpus from scratch and re-checking still returns 2,000
retrieved chunks with 0 unjudged, so the guarantee survives a clean clone. If drift
ever breaks it, the check fails loudly instead of quietly lowering recall.

## Verification

There is **no committed test suite yet**. Behavior has been checked with throwaway
scripts during development, covering: concurrency 1 versus 3 producing identical
selection and rendered output, selection tie-breaking, injected permanent and
transient branch failures, retry classification, citation validation against
hallucinated chunk IDs and paraphrased excerpts, per-paper retrieval scoping,
concurrent Chroma indexing, and interrupt-then-resume without repeating finished
work. Turning these into `tests/` is outstanding work.

No retriever performance or groundedness numbers appear in this README yet. The
figures under Evaluation describe the datasets and their measured pooling bias, and
come from the committed labels and `evals/results/index_coverage.json`. When the
ablation and groundedness runs land, their tables will be generated from
`evals/results/*.json` the same way: any number quoted anywhere comes from committed
output, never from a development run.

## Limitations

- arXiv is the only source; nothing outside it is searched.
- Relevance screening is a model judgement over abstracts and is not yet measured.
- PDF text extraction quality varies, especially for tables, figures, and formulae.
- Citation validation proves an excerpt exists on the cited page. It does not prove
  the excerpt supports the claim built on it.
- arXiv results are not peer reviewed.
- Runs cost model calls: one per candidate screened, plus six per selected paper.
