# arXiv Research Agent — Portfolio Upgrade Plan

## Summary

This is a plan to upgrade the arXiv Research Agent to a production-grade, installable CLI demonstrating credible
LangChain and LangGraph engineering. The finished project will be a production-grade, installable CLI demonstrating
credible LangChain and LangGraph engineering—not a web application.

Success means:

- A fresh checkout can be installed and run from documented commands.
- LangGraph provides native SQLite persistence, fan-out/fan-in, retry handling, and resumability.
- LangChain provides a provider-neutral model interface, structured Pydantic output, caching, token accounting, and optional LangSmith integration.
- Default tests require no API keys or network access.
- The public repository contains a polished README, one verified example, automated tests, and reproducible benchmark tooling.
- Any résumé performance numbers come from committed benchmark output rather than estimates.

Use the display name **arXiv Research Agent**, Python package `arxiv_reviewer`, and console command `arxiv-reviewer`.

## Architecture and Public Interfaces

### Package structure

Refactor the single script into an installable `src/arxiv_reviewer/` package with separate modules for:

- CLI and configuration.
- Pydantic domain/state models.
- LangChain model gateway and usage accounting.
- arXiv search, PDF download, parsing, and caches.
- LangGraph construction, routing, persistence, and report rendering.
- Local benchmark execution.

Use `pyproject.toml` as the dependency and console-entry-point source of truth. Support Python 3.11–3.13 and place development dependencies in a `dev` optional extra.

### CLI contract

Provide these commands:

```text
arxiv-reviewer run --query TEXT
    [--user-query TEXT]                 # Backward-compatible alias
    [--thread-id ID]                    # Generated UUID when omitted
    [--model PROVIDER:MODEL]
    [--max-results 10]
    [--target-papers 4]
    [--max-concurrency 3]
    [--max-paper-tokens 32000]
    [--max-report-tokens 16000]
    [--data-dir .arxiv-reviewer]
    [--output PATH]
    [--no-cache]

arxiv-reviewer resume --thread-id ID
    [--data-dir .arxiv-reviewer]

arxiv-reviewer status --thread-id ID
    [--data-dir .arxiv-reviewer]

arxiv-reviewer benchmark
    [--dataset benchmarks/queries.json]
    [--output benchmarks/results/latest.json]
    [--max-concurrency 3]
    [--compare-concurrency 1,3]
```

Behavior:

- Default model: `google_genai:gemini-3.1-flash-lite`, overridable through `--model` or `ARXIV_REVIEW_MODEL`.
- Initialize models through LangChain’s `init_chat_model`; provider credentials remain provider-standard environment variables.
- Persist the selected model and run settings in graph state, but never persist secrets.
- Print the thread ID and output path immediately when starting a run.
- Reject an existing thread ID on `run`; direct the user to `resume`.
- Refuse to overwrite an unrelated custom output file. Resume may atomically rewrite the output associated with its own thread.
- Exit `0` when a terminal report is produced, including a report with documented partial failures; exit `1` for a resumable fatal failure; exit `2` for invalid arguments or unknown thread IDs.
- `status` must not require a model API key or make network calls.
- Resuming an already completed thread is an idempotent no-op that reports the existing result.

### State and graph

Keep only compact, serializable data in checkpoints:

- Query and run configuration.
- Search queries and normalized paper metadata.
- Relevance outcomes, selected IDs, final analyses, failures, usage records, status, and report text/path.
- Do not store API keys, PDF bytes, or full extracted paper text in graph state.

Use reducer-backed lists for all parallel branch results, then sort them by original arXiv search position before selection or rendering to ensure deterministic output.

Target graph:

```text
START
  │
  ▼
Plan queries + search arXiv
  │
  ├── no candidates ───────────────────────────┐
  ▼                                            │
Fan out relevance evaluation with Send         │
  │                                            │
  ▼                                            │
Select score ≥ 4 papers, highest score first   │
  │                                            │
  ├── none selected ───────────────────────────┤
  ▼                                            │
Fan out PDF analysis with Send                  │
  │                                            │
  └───────────────────────┬────────────────────┘
                          ▼
                 Synthesize Markdown
                          │
                          ▼
                         END
```

Relevance evaluation uses title, authors, date, and abstract. Only selected papers are downloaded and fully analyzed, limiting cost and unnecessary PDF processing.

## Six Implementation Workstreams

### 1. Native SQLite persistence and real resume

- Add `langgraph-checkpoint-sqlite` and compile the graph with `SqliteSaver`.
- Store checkpoints in `.arxiv-reviewer/checkpoints.sqlite`, keyed by LangGraph `thread_id`.
- Remove the custom `checkpointed_node`, JSON serialization helpers, and `--checkpoint` behavior. Existing JSON checkpoints will not be migrated.
- Implement `resume` through an input-less invocation of the existing thread so LangGraph restarts at the last incomplete superstep.
- Use `graph.get_state()` for status reporting, including current/next nodes, candidate counts, successful analyses, failures, and output location.
- Ensure completed writes from successful parallel branches are retained when another branch fails, avoiding repeated model calls after resume.
- Write final reports through a temporary file followed by atomic replacement.
- Follow LangGraph’s [persistence model](https://docs.langchain.com/oss/python/langgraph/persistence), including threads and pending writes.

Acceptance gate: terminate a controlled run after one parallel branch completes, resume it with the same thread ID, and prove the completed branch was not executed again.

### 2. Parallel paper processing

- Replace the `current_paper_index` loop with two explicit map-reduce stages using LangGraph `Send`.
- Stage one evaluates every candidate’s abstract concurrently and returns reducer-backed `CandidateEvaluation` objects.
- A deterministic selection node filters for scores 4–5, sorts by descending score and original search order, and selects at most `target_papers`.
- Stage two downloads, parses, and analyzes selected papers concurrently, returning reducer-backed `AnalysisOutcome` objects.
- Set concurrency through LangGraph’s runtime `max_concurrency`; default to three to avoid aggressive API pressure.
- Keep arXiv query execution sequential with its existing delay because those requests share arXiv rate limits.
- Follow the official [Send/map-reduce pattern](https://docs.langchain.com/oss/python/langgraph/graph-api#send).

Acceptance gate: a fake delayed model must show more than one worker active at concurrency three, while concurrency one remains strictly sequential and both produce identically ordered reports.

### 3. Retries, graceful failures, token limits, and caching

- Apply LangGraph `RetryPolicy` to network/model nodes: three attempts, one-second initial delay, exponential backoff, and jitter.
- Retry timeouts, connection failures, HTTP 408/429/5xx responses, and provider exceptions exposing equivalent status codes. Do not retry validation errors, malformed PDFs, invalid input, or permanent 4xx responses.
- Add per-branch error handlers so exhausted relevance or analysis failures become typed failure outcomes rather than aborting unrelated papers.
- Treat search failure as fatal but resumable. Treat final synthesis failure as recoverable by using the deterministic Markdown renderer.
- Mark the terminal run `partial` and list failures in the report whenever a paper branch fails or deterministic synthesis is used.
- Validate downloaded content before caching: successful response, PDF signature/content type, and non-empty extracted text.
- Cache PDFs and parsed page text under `.arxiv-reviewer/papers/`, keyed by versioned arXiv ID. Use atomic cache writes.
- Configure LangChain’s SQLite LLM cache at `.arxiv-reviewer/llm-cache.sqlite`; its key must include provider/model configuration and prompt/schema content. `--no-cache` disables PDF and LLM caches but never disables checkpoints.
- Use `with_structured_output(..., include_raw=True)` so Pydantic validation and raw usage metadata are both available.
- Record input/output/total token usage when supplied by the provider; represent unavailable values as `null`, never guessed.
- Enforce model-input budgets through `model.get_num_tokens()` with a documented four-characters-per-token fallback.
- If a paper exceeds `--max-paper-tokens`, preserve approximately 60% of the budget from the beginning, 25% from the conclusion/end, and 15% from evenly sampled middle pages. Deduplicate pages and label omitted sections.
- Cap the final synthesis input with `--max-report-tokens`; structured analyses are reduced uniformly if required.
- Use LangGraph’s documented [retry and error-handler lifecycle](https://docs.langchain.com/oss/python/langgraph/fault-tolerance).

Acceptance gate: simulate transient success after retry, permanent per-paper failure, corrupt PDF, oversized paper, cache hit, cache bypass, and report-model failure; every scenario must terminate predictably without losing successful work.

### 4. Automated test suite

Introduce dependency injection around the model, arXiv client, PDF loader, cache, and clock so tests do not patch implementation internals excessively.

Default `pytest` coverage must include:

- Search-plan validation, arXiv normalization, cross-query deduplication, and target limits.
- Graph paths for no candidates, no relevant papers, full success, partial relevance failure, partial analysis failure, and synthesis fallback.
- Parallel reducers and deterministic output ordering.
- Relevance selection and score/search-order tie breaking.
- SQLite checkpoint creation, process-style resume, unknown thread, completed-thread no-op, and retained parallel pending writes.
- Retry classification and maximum-attempt behavior.
- PDF validation, parsing, cache hit/miss/bypass, and atomic writes.
- Token-budget behavior for short, exact-limit, and oversized papers.
- Provider-neutral model initialization and Pydantic structured-output validation.
- Token-usage collection when metadata is present or absent.
- CLI argument validation, exit codes, output collision protection, and all four commands.
- Benchmark aggregation and redaction of secrets.

Quality gates:

- Default tests make zero network calls and require no API key.
- Add `pytest` markers for `live` and `langsmith`; exclude both from the default suite.
- Require at least 85% statement coverage.
- Run Ruff linting and formatting checks.
- Add CI on Python 3.11 and 3.13 that installs `.[dev]`, runs Ruff, tests, coverage, and package build.
- Live tests must skip with a clear reason when credentials are absent.

### 5. README, sample, benchmark, and repository cleanup

Create a polished `README.md` containing:

- One-paragraph product description and concrete use case.
- Technology summary explicitly naming LangChain, LangGraph, Pydantic, Gemini, SQLite, arXiv, PyMuPDF, pytest, and optional LangSmith.
- Mermaid architecture diagram matching the implemented graph.
- Installation, credential setup, quick-start command, resume/status examples, configuration reference, and troubleshooting.
- Explanation of persistence, parallelism, retry policy, caching, token limits, and partial-result behavior.
- Link to one verified example review.
- Test and benchmark commands.
- Limitations: arXiv-only retrieval, LLM judgment fallibility, PDF extraction quality, provider costs, and non-peer-reviewed search results.
- A benchmark table generated from committed machine-readable results, including date, git revision, model, concurrency, cache mode, elapsed time, completion rate, selected-paper yield, branch failure rate, citation-integrity rate, and token usage where available.

Benchmark design:

- Commit five representative research queries spanning agent state tracking, model self-improvement, adversarial reasoning, interpretability, and efficient inference.
- Generate unique thread IDs for each benchmark case.
- Validate report structure and that every cited arXiv ID exists in retrieved metadata.
- Support optional concurrency-one versus concurrency-three comparison with LLM caching disabled; record PDF-cache state separately.
- Never invent or manually adjust measurements. If no live credentials are available, publish the benchmark tooling and explicitly label live results as pending.
- Use metrics from `benchmarks/results/latest.json` as the only source for résumé values such as `X%` or `N queries`.

Repository cleanup:

- Generate a fresh final-pipeline example under `examples/` and manually verify every title, author, URL, and arXiv ID before committing it.
- Remove tracked raw checkpoint JSON, parsed-paper text, and redundant top-level generated reports from the current working tree.
- Do not rewrite Git history.
- Ignore `.arxiv-reviewer/`, SQLite sidecar files, ordinary generated reports, Python/test/build caches, and local environment files.
- Preserve the user’s existing untracked `scratch.txt`; do not overwrite or delete it.
- Add the new Markdown README while preserving the user’s existing deletion of `README.txt`.

Acceptance gate: a reviewer unfamiliar with the repository can install it, run it, understand its architecture, interrupt/resume a run, inspect an example, and reproduce tests solely from the README.

### 6. Optional LangSmith tracing and evaluation

- Keep all local execution functional without a LangSmith account.
- Enable automatic tracing only through standard `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` environment variables.
- Give graph runs, major nodes, thread IDs, model names, and benchmark cases stable trace names/metadata; never attach secrets, PDF bytes, or unnecessary full paper text.
- Reuse the five benchmark cases for an optional LangSmith pytest evaluation.
- Log deterministic feedback for completion, required sections, citation integrity, selected-paper yield, and branch-failure count.
- Offer an optional groundedness/relevance LLM-as-judge evaluator, clearly marked as making additional paid calls.
- Document local dry-run evaluation and tracked evaluation separately.
- Add an `eval` optional dependency extra rather than making hosted evaluation part of the normal installation.
- Follow LangSmith’s [tracing quickstart](https://docs.langchain.com/langsmith/observability-quickstart) and [pytest evaluation workflow](https://docs.langchain.com/langsmith/pytest).

Acceptance gate: ordinary CLI/tests send no LangSmith data; enabling the documented environment variables creates a complete trace associated with the correct thread, and the optional evaluation uploads the benchmark cases and feedback.

## Completion Sequence and Assumptions

Implementation order:

1. Save this document and record the untouched baseline behavior.
2. Add packaging, configuration, typed models, and dependency-injection seams.
3. Replace custom persistence and establish run/resume/status.
4. Implement both parallel map-reduce stages.
5. Add retry/error handling, caches, token controls, and usage accounting.
6. Build deterministic tests and CI; fix failures before documentation.
7. Add benchmark tooling and optional LangSmith evaluation.
8. Generate and verify one new sample, clean generated artifacts, and finish the README.
9. Run the complete offline acceptance suite.
10. If credentials are available, run the live benchmark and publish its unedited output.
11. Derive the final résumé bullet only from implemented features and recorded results.

Defaults chosen because the second preference prompt was unanswered:

- Use an installable `src`-layout package.
- Use deterministic offline tests plus optional paid live benchmarks.
- Curate one final sample and remove raw generated artifacts from the working tree.
- Keep the existing repository remote/name unchanged; remote renaming and CV-site edits are outside this plan.
- Keep the deliverable CLI-only: no FastAPI service, web UI, deployment platform, vector database, or human-approval workflow.
- Use Gemini as the documented default while keeping the model layer provider-neutral.
- Pin compatible dependency ranges in `pyproject.toml` and commit an exact lock generated from the final tested environment.
