# arXiv Research Agent

A LangGraph workflow that searches arXiv, indexes paper text into a vector store,
evaluates papers against a research question, extracts structured notes, and writes
a Markdown literature review with Gemini through LangChain.

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
