# arXiv Research Agent

A LangGraph workflow that searches arXiv, screens candidates against a research
question, indexes the selected papers into a vector store, extracts claims that
cite the pages they came from, and writes a Markdown literature review with Gemini
through LangChain.

## Repository layout

- `arxiv_lit_reviewer.py` — command-line entry point (`run`, `resume`, `status`).
- `arxiv_reviewer/` — application package, split by responsibility.
- `arxiv_reviewer/rag.py` — chunking, embedding, and retrieval over parsed papers.
- `results/reviews/` — historical generated literature reviews.
- `results/parsed/` — extracted paper text retained from development.
- `PORTFOLIO_PLAN.md` — roadmap for the current upgrade.

Run state lives under `.arxiv-reviewer/` (git-ignored): `checkpoints.sqlite` holds
LangGraph checkpoints and `chroma/` holds the persistent vector index.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in a `.env` file at the repository root.

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

Interrupted runs resume from the last checkpoint. Branches that already finished
are not re-executed, so their downloads and model calls are not paid for twice.

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
