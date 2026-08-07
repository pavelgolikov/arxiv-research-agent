# arXiv Research Agent — Portfolio Upgrade (1-Week Focused Plan)

> Supersedes `portfolio_core_upgrade_plan.md` and `portfolio_upgrade_plan.md`.

## Context

The repo is a working but entry-level LangGraph prototype (~750 lines) meant to demonstrate
LangGraph + LangChain + RAG to employers. Three gaps make it unconvincing as a portfolio piece:

1. **There is no RAG.** No embeddings, no vector store, no chunking, no retriever, no reranking.
   [analysis.py:82](arxiv_reviewer/analysis.py#L82) puts the *entire* paper text into one prompt —
   the opposite of retrieval-augmented generation.
2. **LangGraph usage is tutorial-level.** A manual `current_paper_index` loop
   ([analysis.py:56](arxiv_reviewer/analysis.py#L56)) instead of `Send` fan-out; no reducers, no
   native checkpointer, no retry policy. Worse, `checkpointed_node`
   ([checkpointing.py:51](arxiv_reviewer/checkpointing.py#L51)) writes state to JSON after every
   node and **nothing ever reads it back** — `run_reviewer` always starts from a fresh dict, so the
   headline "checkpointing" feature cannot actually resume.
3. **Zero evaluation.** This is the real miss, given a background in agent evaluation (ArbiGraph,
   the adversarial robust-reasoning benchmark, Vector Institute). "Wired up a LangGraph pipeline" is
   a bootcamp-level claim; "built a grounded RAG agent and measured retrieval recall, citation
   integrity, and groundedness against a labeled set" is not.

The two prior plan docs are rigorous but aimed wrong: they optimize for production polish
(packaging, CI, 85% coverage, clean-install matrices) and put "vector database or persistent
semantic search index" under **Explicitly excluded** — dropping the exact capability the project
exists to demonstrate. This plan re-aims at RAG + evaluation.

**Target outcome:** in ~1 focused week, a repo whose README truthfully says *grounded hybrid-RAG
research agent with a measured retrieval ablation*, aimed at AI research engineer/scientist roles.

**Deliberately out of scope** (stretch goals, listed at the end): LangSmith, human-in-the-loop
`interrupt`, FastAPI/UI/Docker, `pyproject.toml` packaging, CI, coverage-percentage targets.

---

## Architecture after the change

```text
START
  │
  ▼  plan_search      LLM → 1-3 arXiv queries (SearchPlan)   [exists, keep]
  ▼  search_arxiv     sequential, rate-limited, dedup by canonical ID
  │
  ├── no candidates ──────────────────────────────────────────────┐
  ▼                                                                │
  ⇉  screen_candidate   Send fan-out, ABSTRACT ONLY (no PDF yet)   │
  ▼  select_papers      deterministic: score desc, then search pos │
  │                                                                │
  ├── none selected ──────────────────────────────────────────────┤
  ▼                                                                │
  ⇉  ingest_paper       Send fan-out: download → page-preserving   │
  │                     parse → chunk → embed → Chroma             │
  ▼  build_retriever    hybrid (Chroma dense + BM25) → RRF → rerank │
  ⇉  analyze_paper      Send fan-out: per-facet retrieval,         │
  │                     claims + evidence chunk IDs                │
  ▼  validate_evidence  every citation checked against real chunks │
  ▼  synthesize         corpus-level retrieval across ALL papers   │
  │                     (themes/gaps) → structured output          │
  ├── synthesis fails ──► deterministic renderer, mark `partial`   │
  ▼  render_report      Markdown written atomically ◄──────────────┘
  ▼
 END
```

The key shift: **retrieval becomes load-bearing.** Facet-level and corpus-level questions are
answered from retrieved chunks, not from a full-text dump. This is what makes the RAG claim honest,
and it simultaneously cuts token cost and makes page-level citations possible.

---

## Workstream 1 — Real RAG (the headline)

New module `arxiv_reviewer/rag.py`, replacing the full-text prompting in
[analysis.py](arxiv_reviewer/analysis.py).

**Ingestion.** Modify `download_parse_node` in [retrieval.py:109](arxiv_reviewer/retrieval.py#L109)
to preserve page boundaries (it currently joins all pages with `\n`, destroying the page numbers
citations need). Emit `ParsedPage(page_number, text)` records. Chunk with
`RecursiveCharacterTextSplitter` (~1000 chars, 150 overlap), carrying
`{arxiv_id, page_number, chunk_id, section?}` metadata on every chunk.

**Index.** Chroma (`langchain-chroma`) persisted at `.arxiv-reviewer/chroma/`, one collection per
run thread. Embeddings via `GoogleGenerativeAIEmbeddings` (already have `langchain-google-genai`,
so no new provider key).

**Retrieval — hybrid + rerank.**

- Dense: Chroma `as_retriever()`.
- Sparse: `BM25Retriever` (`langchain-community` + `rank_bm25`).
- Fusion: `EnsembleRetriever` (reciprocal rank fusion), fetch ~20.
- Rerank: `ContextualCompressionRetriever` + `CrossEncoderReranker` with
  `HuggingFaceCrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB, CPU-fine) → top 5.
- Query expansion: multi-query — expand each facet question into 2-3 paraphrases before retrieval.

Expose `--retriever {dense,bm25,hybrid,hybrid-rerank}` (default `hybrid-rerank`). **This flag is not
decoration** — it is the ablation axis for Workstream 3, and it lets the offline test suite run
without downloading the cross-encoder.

**Grounded analysis.** For each selected paper, run one retrieval per facet (research problem,
method, experimental setup, findings, limitations) instead of one giant prompt. The model returns
`SupportedClaim(text, evidence_ids)`; `EvidenceRef` carries `chunk_id`, `arxiv_id`, `page_number`,
and a short excerpt.

**Validation (deterministic, no LLM).** A citation is valid only if: the chunk ID exists, it belongs
to the paper under analysis, and the normalized excerpt actually occurs in that chunk's text. Invalid
claims are dropped and the analysis is marked partial. This is cheap, fully testable, and it is the
anti-hallucination story that makes the project interesting to an evaluation-focused reader.

**Note on imports:** LangChain has shuffled these classes between `langchain`,
`langchain-community`, and `langchain-huggingface` across versions. Verify every import path against
the installed version at implementation time rather than trusting the names above.

## Workstream 2 — LangGraph, done properly

In [workflow.py](arxiv_reviewer/workflow.py) and [analysis.py](arxiv_reviewer/analysis.py):

- **Delete the index loop.** Remove `advance_paper_node`, `route_after_advance_paper`, and
  `current_paper_index`. Replace with two `Send` map-reduce stages (screen, analyze). Results land in
  reducer-backed lists (`Annotated[list[...], operator.add]`), then a deterministic sort by
  (score desc, original search position) before selection and rendering — so concurrency never
  changes the output.
- **Screen on abstracts before downloading.** Today every candidate PDF is downloaded before
  relevance is judged. Screening from metadata + abstract first eliminates most downloads and most
  cost, and it is a visible design improvement to talk about.
- **Native persistence.** Add `langgraph-checkpoint-sqlite`, compile with `SqliteSaver` at
  `.arxiv-reviewer/checkpoints.sqlite`, key every invocation by `thread_id`. **Delete
  [checkpointing.py](arxiv_reviewer/checkpointing.py) entirely** and the `--checkpoint` flag. Keep
  PDF bytes and full paper text *out* of state — they live in the Chroma index and file cache.
- **CLI: `run` / `resume` / `status`.** Restructure
  [arxiv_lit_reviewer.py](arxiv_lit_reviewer.py) into subcommands; `run` prints the thread ID
  up front, `resume` does an input-less invocation on that thread, `status` reads
  `graph.get_state()` and must work with no API key and no network. Also drop the
  `from arxiv_reviewer import *` wildcard at [arxiv_lit_reviewer.py:12](arxiv_lit_reviewer.py#L12).
- **`RetryPolicy`** on network/model nodes: 3 attempts, exponential backoff with jitter. Retry
  timeouts/429/5xx; never retry validation errors or malformed PDFs. A failed branch becomes a typed
  failure outcome; siblings still finish and the report is marked `partial`.
- **Fix the client churn** in [gemini_client.py:25](arxiv_reviewer/gemini_client.py#L25) — a new
  `ChatGoogleGenerativeAI` is constructed on every single call. Cache it, and initialize through
  `init_chat_model("google_genai:...")` so the model layer is provider-neutral (`--model` flag).

## Workstream 3 — Evaluation (the differentiator)

New `evals/` directory. All datasets **frozen and committed** so results are reproducible and don't
drift with live arXiv ranking.

**A. Retrieval ablation** — the centerpiece. Committed corpus of parsed chunks from ~5 papers
(one is already on disk: `results/parsed/2503.21676v2_parsed.txt`). Hand-label ~40-50
question→relevant-chunk pairs. Report **recall@5, MRR, nDCG@10** for four configurations:
`dense` / `bm25` / `hybrid` / `hybrid+rerank`.

*Why this is the most valuable artifact in the project:*

The RAG stack has four stacked components, each of which is a claim. Building them proves nothing —
a broken reranker and a good one look identical in a README. An ablation in the ordinary ML sense
(same fixed queries, one component added at a time, measure) converts unfalsifiable claims into
measurements a reader can interrogate.

*Ground truth:* for ~40-50 questions the pipeline actually asks ("what is the research problem?",
"what dataset was this evaluated on?", "what limitations do the authors list?"), read the paper and
mark which specific chunks genuinely contain the answer. Frozen and committed.

*Metrics, in this setting:*

- **recall@5** — of the chunks that truly answer the question, what fraction reach the top 5 handed
  to the LLM. The one that matters most: **retrieval recall is a hard ceiling on answer quality**,
  since the model cannot ground a claim in a chunk it never saw.
- **MRR** — where the *first* correct chunk lands (1/rank). Catches "retrieved it but buried it."
- **nDCG@10** — graded, position-discounted; rewards ranking the most relevant chunks highest.

*Output:* one table generated from `evals/results/retrieval.json`, four rows (dense / bm25 / hybrid /
hybrid+rerank) × three metrics.

*A negative result is a fine result.* If reranking buys nothing measurable, publish that —
"the cross-encoder did not improve recall@5, likely because the top-20 candidates were already
saturated" is more credible than a suspiciously clean win. The only failure mode is labeling the
data and then not publishing what it says.

**B. Screening quality.** ~5 queries × frozen candidate metadata, labeled `0` irrelevant /
`1` related / `2` central. Report precision@target and central-paper recall. Use this to *choose*
the relevance threshold rather than guessing it — and say so in the README.

**C. Groundedness.** Run on generated reports: citation referential integrity (every cited chunk
exists, belongs to the right paper, excerpt verifiable — deterministic, no LLM needed) and
claim-support rate. Optional LLM-judge faithfulness score, clearly labeled as an extra paid call.

Results write to `evals/results/*.json` (machine-readable, unedited). **The README table and every
externally quoted number are generated from these files.** If a number isn't in a committed results
file, it doesn't get quoted anywhere — carried over from the prior plan docs.

## Workstream 4 — Tests, README, cleanup

**Tests** (`tests/`, pytest, no network, no API key — inject fakes for model/arXiv/HTTP/embeddings):
graph terminal paths (no candidates / none selected / full success / partial branch failure /
synthesis fallback); reducer determinism at concurrency 1 vs 3; evidence validation (valid, wrong
paper, missing excerpt, bad chunk ID); SQLite resume (interrupt → resume → completed branch not
re-run); retry classification. Target roughly 25-35 meaningful tests. **No coverage percentage
target** — chasing 85% is where the week disappears.

**README** — assume a reader who reads this and nothing else:

- One-paragraph description + the architecture diagram above (as Mermaid).
- Explicit tech line: LangGraph, LangChain, Chroma, BM25, cross-encoder reranking, Gemini, Pydantic,
  SQLite, PyMuPDF, pytest.
- **The retrieval ablation table**, generated from `evals/results/`.
- How grounding/citation validation works, and honest limitations (arXiv-only, PDF extraction
  quality, LLM fallibility, non-peer-reviewed sources).
- Link to one example report whose citations have been **manually verified** end to end.

**Cleanup:** delete the 7 committed checkpoint JSONs, the 7 near-duplicate reviews, and the raw
parsed text from the working tree (keep one verified example under `examples/`); gitignore
`.arxiv-reviewer/`. Commit the already-pending deletion of the two old plan docs — this file
replaces them. Don't rewrite git history.

---

## Suggested day-by-day

| Day | Work |
| --- | --- |
| 1 | Page-preserving parse, chunking, Chroma ingest, dense retrieval end to end. Delete `checkpointing.py`, wire `SqliteSaver` + `run`/`resume`/`status`. |
| 2 | BM25 + `EnsembleRetriever` + cross-encoder rerank + multi-query. `--retriever` flag. |
| 3 | Rewrite analysis as per-facet grounded retrieval; `SupportedClaim`/`EvidenceRef`; deterministic validation. |
| 4 | `Send` fan-out for screening + analysis, reducers, deterministic sort, `RetryPolicy`, partial-failure handling. |
| 5 | Label the retrieval and screening datasets. **Budget the full day** — hand-labeling is the real cost and the thing that makes the project credible. |
| 6 | Eval runners, ablation table, groundedness metrics, commit results JSON. |
| 7 | Tests, README, verified example, repo cleanup, final live run. |

## Verification

- `pytest` — passes with no network and no `GEMINI_API_KEY` set (block outbound calls via fixture).
- `python arxiv_lit_reviewer.py run --query "..."` → note the thread ID; `Ctrl-C` mid-analysis;
  `... status --thread-id X` shows partial state; `... resume --thread-id X` completes **without
  re-running the already-finished branches** (check via a call counter or log).
- `... run --retriever dense` vs `--retriever hybrid-rerank` on the same query → different retrieved
  context, same report structure.
- `python -m evals.run_retrieval` → writes `evals/results/retrieval.json`; the four configurations
  produce distinguishable recall@5 / nDCG@10.
- Open the committed example report and manually confirm every arXiv ID, link, page number, and
  evidence excerpt against the real paper. Do this before claiming "verified" anywhere.
- `... status` in a shell with all provider credentials unset → succeeds.

## Summary claim (fill the brackets from `evals/results/`, don't estimate)

> Grounded literature-review agent (LangGraph + LangChain) over arXiv: hybrid dense+BM25 retrieval
> with cross-encoder reranking across a persistent Chroma index, page-level citation validation, and
> parallel map-reduce paper analysis with SQLite-backed checkpointing and resume. Hand-labeled a
> retrieval benchmark and ran a four-configuration ablation showing
> **[+X pts recall@5 / +Y nDCG@10]** from hybrid retrieval + reranking over dense-only, with
> **[Z]%** citation referential integrity across generated reports.

## Stretch (only after the week lands)

LangSmith tracing + hosted evals · `interrupt` for human-in-the-loop paper approval (a genuinely
strong LangGraph capability) · `pyproject.toml` + console entry point + CI · FastAPI streaming
endpoint and a small UI, if a clickable demo becomes worth the time.
