# arXiv Research Agent

A LangGraph workflow that searches arXiv, screens candidates against a research
question, indexes the selected papers into a vector store, extracts claims that
cite the pages they came from, and writes a Markdown literature review with Gemini
through LangChain.

## Project status

The pipeline runs end to end. The evaluation half of the project — the part that
measures whether the retrieval stack actually helps — is not built yet.

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

**Not built yet**

| Area | Notes |
| --- | --- |
| Labeled retrieval dataset | ~40-50 hand-labeled question to relevant-chunk pairs, frozen and committed |
| Labeled screening dataset | ~5 queries of candidate metadata labeled irrelevant / related / central |
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

## Verification

There is **no committed test suite yet**. Behavior has been checked with throwaway
scripts during development, covering: concurrency 1 versus 3 producing identical
selection and rendered output, selection tie-breaking, injected permanent and
transient branch failures, retry classification, citation validation against
hallucinated chunk IDs and paraphrased excerpts, per-paper retrieval scoping,
concurrent Chroma indexing, and interrupt-then-resume without repeating finished
work. Turning these into `tests/` is outstanding work.

No performance or quality numbers appear in this README on purpose. Once
`evals/results/*.json` exists, the ablation table and groundedness figures will be
generated from it, and any number quoted anywhere will come from that committed
output rather than from a development run.

## Limitations

- arXiv is the only source; nothing outside it is searched.
- Relevance screening is a model judgement over abstracts and is not yet measured.
- PDF text extraction quality varies, especially for tables, figures, and formulae.
- Citation validation proves an excerpt exists on the cited page. It does not prove
  the excerpt supports the claim built on it.
- arXiv results are not peer reviewed.
- Runs cost model calls: one per candidate screened, plus six per selected paper.
