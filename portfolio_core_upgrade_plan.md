# arXiv Research Agent — Core Portfolio Upgrade Plan

## 1. Objective

Upgrade the existing arXiv literature-review prototype into a robust, local-first research agent that demonstrates credible LangGraph and LangChain engineering while producing evidence-grounded research reports.

The project remains a command-line application. The core work focuses on:

- Native persistence and genuine resume behavior.
- Parallel, deterministic paper processing.
- Evidence-grounded analysis and report generation.
- Relevance-quality evaluation.
- Predictable retries, partial failures, caching, and resource limits.
- Automated offline tests and reproducible benchmarks.
- A polished, verified example and clear documentation.
- Packaging and clean-install verification only after the rest of the core system is complete.

Use the display name **arXiv Research Agent**, Python package name `arxiv_reviewer`, and final console command `arxiv-reviewer`.

Do not describe the result as universally “production-grade.” Describe it as a robust, local-first research agent with production-style reliability features.

## 2. Core Definition of Done

The core upgrade is complete when all of the following are true:

- A user can start, inspect, interrupt, and resume a run using a stable thread ID.
- LangGraph uses native SQLite checkpoints rather than custom JSON snapshots.
- Candidate relevance evaluation and selected-paper analysis run concurrently with configurable limits.
- Parallel results remain deterministically ordered.
- Relevance screening happens from metadata and abstracts before PDFs are downloaded.
- Important report claims contain validated evidence references to paper pages.
- A failed paper branch does not discard successful branches or prevent a partial report.
- Search failures remain resumable, and final-synthesis failures use a deterministic fallback.
- PDFs and parsed pages are cached safely and are subject to size, page, and token limits.
- Default tests require no API keys and make no network calls.
- A small committed relevance dataset measures screening quality, not just code correctness.
- Benchmark output is machine-readable and is the only source of résumé performance claims.
- One final example report has been manually checked against its cited papers.
- Only after all preceding goals pass, the project is packaged, installed in clean environments, and exposed through the final console command.

## 3. Scope Boundaries

### Included

- arXiv query planning and search.
- Abstract-based relevance ranking.
- PDF download and page-preserving text extraction.
- Structured paper analysis with evidence references.
- Cross-paper synthesis and deterministic Markdown rendering.
- SQLite persistence, status reporting, and resume.
- Parallel map-reduce stages.
- Retry classification, error handling, and partial results.
- PDF/parsed-text caching and model usage accounting.
- Offline tests, a small quality evaluation set, and live benchmark tooling.
- Documentation, example verification, repository cleanup, and final packaging.

### Explicitly excluded

- Web UI, FastAPI service, or hosted deployment.
- Vector database or persistent semantic search index.
- Retrieval outside arXiv.
- Human approval workflows.
- Multi-user or distributed execution.
- LangSmith tracing or hosted evaluation.
- SQLite LLM-response caching.
- Live verification across multiple model providers.
- Migration of historical JSON checkpoints.
- Automated CV-site editing or GitHub repository renaming.

The model gateway should use LangChain’s provider-neutral interface, but the core deliverable only needs to be exercised live with the documented Gemini default and tested offline with fake models.

## 4. Intended Architecture

```text
START
  |
  v
Validate configuration and create run record
  |
  v
Generate search queries and search arXiv sequentially
  |
  +---- no candidates -------------------------------+
  |                                                  |
  v                                                  |
Fan out abstract relevance evaluations               |
  |                                                  |
  v                                                  |
Deterministically rank and select candidates          |
  |                                                  |
  +---- none selected -------------------------------+
  |
  v
Fan out PDF download, parsing, and grounded analysis
  |
  v
Validate evidence references
  |
  v
Generate structured cross-paper synthesis
  |
  +---- synthesis failure --> deterministic fallback
  |
  v
Render Markdown and atomically write report
  |
  v
END
```

The graph should contain only compact, serializable checkpoint state. PDF bytes and full extracted paper text belong in the filesystem cache, not graph state.

## 5. Public Command Contract

Before final packaging, commands may be exercised through the existing launcher:

```text
python arxiv_lit_reviewer.py run --query TEXT
python arxiv_lit_reviewer.py resume --thread-id ID
python arxiv_lit_reviewer.py status --thread-id ID
python arxiv_lit_reviewer.py benchmark
```

After the final packaging milestone, the same commands must be available as:

```text
arxiv-reviewer run --query TEXT
arxiv-reviewer resume --thread-id ID
arxiv-reviewer status --thread-id ID
arxiv-reviewer benchmark
```

### `run`

```text
run --query TEXT
    [--user-query TEXT]
    [--thread-id ID]
    [--model PROVIDER:MODEL]
    [--max-results 10]
    [--target-papers 4]
    [--max-concurrency 3]
    [--max-paper-tokens 32000]
    [--max-report-tokens 16000]
    [--max-pdf-mb 50]
    [--data-dir .arxiv-reviewer]
    [--output PATH]
    [--no-cache]
```

Behavior:

- Generate a UUID thread ID when none is supplied.
- Print the thread ID and intended output path before expensive work begins.
- Reject an existing thread ID and direct the user to `resume`.
- Refuse to overwrite an unrelated custom output file.
- Read the default model from `ARXIV_REVIEW_MODEL`, falling back to `google_genai:gemini-3.1-flash-lite`.
- Never persist API keys or other secrets.

### `resume`

```text
resume --thread-id ID [--data-dir .arxiv-reviewer]
```

Behavior:

- Resume through an inputless graph invocation using the original thread configuration.
- Reject unknown thread IDs.
- Treat an already completed thread as an idempotent no-op and report the existing output.
- Reuse successful pending writes so completed parallel branches are not called again.

### `status`

```text
status --thread-id ID [--data-dir .arxiv-reviewer]
```

Report:

- Overall status: `pending`, `running`, `partial`, `failed`, or `complete`.
- Current and next graph nodes.
- Candidate and selected-paper counts.
- Successful and failed relevance branches.
- Successful and failed analysis branches.
- Recorded token usage when available.
- Report path.

`status` must not initialize a model, require an API key, download a PDF, or make a network call.

### Exit codes

- `0`: a complete or explicitly partial terminal report exists.
- `1`: execution failed but the thread is resumable.
- `2`: invalid arguments, output collision, or unknown thread ID.

## 6. Core Data Models

Use Pydantic for runtime-validated domain and model-output objects. Keep the LangGraph state as a compact typed mapping with reducer-backed parallel result lists.

### Run configuration

Include:

- Query.
- Model identifier.
- Search and selection limits.
- Concurrency limit.
- PDF, paper-input, and report-input limits.
- Cache setting.
- Data and output paths.

### Paper identity and metadata

Track both:

- Canonical/base arXiv ID for deduplication.
- Versioned arXiv ID for reproducible caching and citation.

Metadata includes title, authors, abstract, publication/update dates, entry URL, PDF URL, and original search position.

### Candidate evaluation

Include:

- Paper ID and original search position.
- Relevance score.
- Concise reason.
- Outcome status.
- Failure details when evaluation did not succeed.
- Model usage metadata when available.

### Parsed page

Include:

- One-indexed PDF page number.
- Extracted text.
- Optional detected section heading.

Parsed pages remain in the filesystem cache and must not be copied wholesale into checkpoints.

### Evidence reference

Include:

- Stable evidence ID.
- Versioned arXiv ID.
- Page number.
- Detected section when available.
- Short supporting excerpt.

An evidence reference is valid only if:

- Its paper is the paper being analyzed.
- Its page number exists.
- Its normalized excerpt occurs on that extracted page.
- Its excerpt stays within a configured length limit.

### Supported claim

Include:

- Claim text.
- One or more evidence IDs.

Research problem, method, experimental setup, primary findings, and limitations should be represented as supported claims rather than unreferenced prose.

### Analysis outcome

Include either:

- A successful structured analysis with validated evidence and usage data; or
- A typed failure with stage, category, retryability, and safe error message.

### Cross-paper synthesis

Represent overview statements, themes, gaps, comparisons, and reading-order explanations as structured objects. Any factual synthesis statement must reference one or more validated paper evidence IDs.

## 7. Milestone 1 — Baseline Safety Net and Test Seams

### Goals

- Record current intended behavior before changing the graph.
- Establish dependency-injection boundaries.
- Begin testing immediately rather than waiting until the refactor is finished.

### Work

- Add tests for current query validation, metadata normalization, deduplication, relevance routing, report fallback, and output writing.
- Introduce interfaces or protocols for:
  - Chat model creation and invocation.
  - arXiv search.
  - HTTP/PDF retrieval.
  - PDF parsing.
  - Cache access.
  - Clock/sleep behavior.
  - UUID generation where deterministic tests need it.
- Create fake implementations that require no credentials or network.
- Add a network-blocking test fixture so unexpected outbound calls fail the default suite.
- Stop importing the entire package with a wildcard from the launcher.
- Ensure CLI argument parsing and `--help` can run without importing PDF or model-provider integrations.

### Acceptance gate

- Baseline tests pass without an API key or network.
- The fakes can drive a complete small workflow.
- `--help` works even if optional runtime integrations are unavailable.

## 8. Milestone 2 — Compact State, Native Persistence, and Resume

### Goals

- Replace custom JSON snapshots with LangGraph-native SQLite persistence.
- Establish real `run`, `resume`, and `status` behavior before introducing parallelism.

### Work

- Define the compact graph state and typed run statuses.
- Remove `checkpointed_node`, JSON checkpoint serialization, and `--checkpoint` behavior.
- Add `langgraph-checkpoint-sqlite` and use the appropriate SQLite saver for the graph’s execution style.
- Store the database at `.arxiv-reviewer/checkpoints.sqlite` by default.
- Key every graph invocation with a LangGraph `thread_id`.
- Keep model name and non-secret settings in state.
- Keep PDF bytes and full parsed text out of state.
- Implement `run`, `resume`, and `status` through the existing Python launcher.
- Use graph state inspection for status rather than maintaining a second status database.
- Write reports through a temporary file followed by atomic replacement.
- Make completed-thread resume an idempotent no-op.

### Required tests

- New thread creation.
- Duplicate thread rejection.
- Unknown-thread status and resume.
- State inspection without a model key.
- Resume from an incomplete sequential graph.
- Completed-thread no-op.
- Compact checkpoints contain no PDF bytes, full paper text, or secrets.
- Atomic report replacement and output collision protection.

### Acceptance gate

- A controlled interrupted run resumes from its last completed graph step.
- Status works in a process with all provider credentials removed.

## 9. Milestone 3 — Abstract-First Relevance and Parallel Map-Reduce

### Goals

- Avoid downloading clearly irrelevant papers.
- Replace the sequential paper loop with deterministic parallel stages.

### Work

- Generate one to three validated search queries.
- Execute arXiv queries sequentially with rate-limit-aware delays.
- Normalize metadata and deduplicate by canonical arXiv ID.
- Preserve original search order.
- Fan out candidate evaluations with LangGraph `Send`.
- Evaluate relevance using title, authors, dates, and abstract only.
- Store candidate outcomes in a reducer-backed list.
- Rank successful candidates by relevance score, then original search position.
- Select at most `target_papers`.
- Add a documented top-ranked fallback if a strict relevance threshold would otherwise produce too few papers; finalize the rule using the relevance evaluation dataset.
- Fan out selected-paper processing with `Send`.
- Apply runtime `max_concurrency`, defaulting to three.
- Sort all reduced results before rendering or benchmarking.

### Required tests

- Query-plan validation and fallback.
- Cross-query canonical-ID deduplication.
- Empty search result path.
- Partial relevance failure.
- No-relevant-paper path.
- Score and search-order tie breaking.
- Target-paper limit.
- More than one fake worker active at concurrency three.
- Strictly sequential behavior at concurrency one.
- Identical final ordering at concurrency one and three.

### Acceptance gate

- Only selected papers reach the PDF-processing stage.
- Parallel and sequential executions produce identically ordered results.

## 10. Milestone 4 — Grounded PDF Analysis and Deterministic Reporting

### Goals

- Make the research output auditable.
- Prevent free-form synthesis from inventing unsupported claims or citations.

### Work

- Preserve page boundaries during PDF extraction.
- Add lightweight section-heading detection where extraction quality permits.
- Ask the analysis model for supported claims and evidence references.
- Validate every evidence reference against cached extracted pages.
- Reject invalid page numbers and excerpts that cannot be located.
- Mark an analysis partial when some claims fail validation; fail the branch if no useful grounded analysis remains.
- Bound evidence excerpt length so checkpoints and reports remain compact.
- Generate cross-paper synthesis as structured Pydantic output rather than free-form Markdown.
- Require evidence IDs for factual overview, theme, comparison, and gap statements.
- Validate synthesis evidence IDs before rendering.
- Render the final Markdown deterministically from validated objects.
- Use readable citations containing title/arXiv link and page number.
- If cross-paper synthesis fails, render the validated per-paper analyses and mark the report partial.

### Report sections

- Search summary.
- Method and limitations notice.
- Overview.
- Selected papers.
- Comparison table.
- Research themes.
- Research gaps.
- Suggested reading order.
- Failures and partial-result notice when applicable.
- Usage summary when available.

### Required tests

- Valid evidence excerpt and page.
- Missing excerpt.
- Out-of-range page.
- Evidence referencing the wrong paper.
- Duplicate evidence IDs.
- Partially grounded paper analysis.
- Completely ungrounded analysis failure.
- Invalid synthesis reference.
- Deterministic report ordering.
- Synthesis fallback.
- Markdown escaping and valid arXiv links.

### Acceptance gate

- Every evidence-required claim in the final report has a validated evidence reference.
- The renderer cannot introduce a paper or evidence ID absent from validated state.

## 11. Milestone 5 — Reliability, Resource Controls, and Caching

### Goals

- Make failures predictable and preserve successful work.
- Prevent unexpectedly large PDFs or prompts from exhausting resources.

### Retry and failure behavior

- Apply LangGraph retry policies to network and model nodes.
- Use three attempts including the first, exponential backoff, jitter, and a one-second initial retry interval.
- Retry timeouts, connection failures, HTTP 408/429/5xx responses, and equivalent provider errors.
- Do not retry invalid input, Pydantic validation errors, malformed PDFs, unsupported content, or permanent 4xx responses.
- Use node error handlers after retries are exhausted.
- Convert relevance and analysis branch failures into typed outcomes.
- Treat search failure as fatal but resumable.
- Treat synthesis failure as recoverable through deterministic rendering.
- Mark terminal output `partial` when any selected branch fails or synthesis falls back.

### PDF safety

- Stream downloads rather than loading unbounded responses blindly.
- Enforce a configurable maximum download size, defaulting to 50 MiB.
- Validate successful status, PDF signature, parseability, positive page count, and non-empty extracted text.
- Reject oversized, corrupt, encrypted/unreadable, or empty documents with typed errors.
- Use explicit connection and read timeouts.

### PDF and parsed-page cache

- Cache validated PDFs and extracted page text under `.arxiv-reviewer/papers/`.
- Key cache entries by versioned arXiv ID.
- Write cache files atomically.
- Never cache failed or partially downloaded content.
- Make `--no-cache` bypass PDF and parsed-page caches without disabling checkpoints.
- Do not add LLM-response caching in the core implementation.

### Token budgets

- Count tokens with the model tokenizer when available.
- Use a documented four-characters-per-token estimate only when token counting is unavailable.
- Enforce `max-paper-tokens` before paper-analysis calls.
- Prefer section-aware selection: abstract/introduction, methods/results, and conclusion/limitations.
- Fill unused budget with evenly sampled middle pages.
- Label omitted page ranges and retain original page numbers.
- Enforce `max-report-tokens` before cross-paper synthesis.
- Reduce structured analyses uniformly when necessary rather than silently truncating the last papers.

### Usage accounting

- Invoke structured models with raw output metadata available.
- Record input, output, and total tokens when supplied.
- Store unavailable values as `null`; never estimate them for benchmark claims.
- Aggregate usage by stage and complete run.

### Graceful interruption

- Handle normal interruption signals by requesting a checkpoint-safe drain when supported.
- Print the thread ID and resume command on interruption.
- Document that a second forced termination may stop immediately.

### Required tests

- Transient success after retry.
- Maximum retry exhaustion.
- Permanent error without retry.
- One parallel branch failing while others complete.
- Successful pending writes retained on resume.
- Oversized response with and without `Content-Length`.
- Invalid PDF signature, corrupt PDF, and empty extraction.
- Cache hit, miss, corrupt entry, bypass, and atomic write.
- Short, exact-limit, and oversized paper inputs.
- Missing and present token metadata.
- Graceful interruption leaves a resumable thread.

### Acceptance gate

- Every simulated failure has a deterministic status, exit code, and report effect.
- A successful branch is not repeated after a sibling failure and resume.

## 12. Milestone 6 — Quality Evaluation, Benchmark, Documentation, and Cleanup

### Relevance evaluation dataset

Commit a small, manually labeled dataset containing candidate metadata for representative queries. Use labels such as:

- `0`: irrelevant.
- `1`: related but not central.
- `2`: central to the query.

Requirements:

- Cover at least five research queries relevant to the portfolio.
- Include both obvious and difficult negative examples.
- Store the label and a short human rationale.
- Freeze candidate metadata so results do not change with live arXiv ranking.
- Keep evaluation separate from deterministic unit tests.

Report ranking and screening metrics such as central-paper recall at the target cutoff, precision of selected papers, and ranking quality. Choose and document the runtime selection threshold/fallback after examining these results. Do not tune against unrecorded examples.

### Offline tests

- Default `pytest` must make zero network calls and require no credentials.
- Add offline coverage for every graph terminal path.
- Include reducer, ordering, resume, retry, cache, evidence, rendering, benchmark, and CLI behavior.
- Require at least 85% statement coverage while prioritizing meaningful branch coverage over metric gaming.
- Run Ruff lint and formatting checks.

### Live benchmark tooling

Commit five representative queries spanning:

- Agent state tracking.
- Model self-improvement.
- Adversarial reasoning.
- Mechanistic interpretability.
- Efficient inference.

For each benchmark case, record:

- Date and git revision.
- Query and unique thread ID.
- Model and run configuration.
- Cache state.
- Elapsed time.
- Candidate and selected-paper counts.
- Completion status.
- Branch failure counts.
- Grounded-claim coverage.
- Citation referential-integrity rate.
- Token usage when provider metadata is available.

Write unedited machine-readable results under `benchmarks/results/`. Any README or résumé numbers must be generated from this output. If live credentials are unavailable, commit the tooling and label live results as pending.

### Verified example

- Generate one report with the final pipeline.
- Manually verify paper titles, authors, arXiv IDs, URLs, cited pages, and evidence excerpts.
- Check a sample of synthesized claims against the cited evidence.
- Commit the report under `examples/` with a short verification note and generation metadata.
- Do not present model-generated output as manually verified until this review is complete.

### Documentation

Prepare the README content before packaging, leaving final installation commands for Milestone 7. Include:

- Product description and concrete use case.
- Architecture diagram matching the implemented graph.
- Explanation of grounding, persistence, resume, parallelism, retries, caching, token limits, and partial reports.
- Command reference for the launcher.
- Status and resume examples.
- Test and benchmark commands.
- Link to the verified example.
- Benchmark table generated from machine-readable results.
- Limitations: arXiv-only search, relevance-model fallibility, PDF extraction errors, evidence-validation limits, model cost, and non-peer-reviewed content.

### Repository cleanup

- Remove tracked raw JSON checkpoints and parsed-paper text from the working tree.
- Retain only the final verified example rather than multiple redundant generated reviews.
- Ignore `.arxiv-reviewer/`, SQLite sidecars, generated reports, caches, virtual environments, and build artifacts.
- Do not rewrite Git history.

### Acceptance gate

- The full offline suite, quality evaluation, and benchmark aggregation run from documented repository commands.
- The verified example contains no invalid paper IDs, links, pages, or excerpts.
- The README accurately describes implemented behavior and does not claim unrecorded results.

## 13. Milestone 7 — Packaging and Clean Installation

This is intentionally the final implementation milestone. Do not begin the packaging migration until Milestones 1–6 pass their acceptance gates.

### Goals

- Convert the completed core application into an installable project.
- Prove installation and execution from clean environments.
- Avoid allowing packaging work to distract from the research and reliability behavior earlier in the project.

### Work

- Move the completed package to `src/arxiv_reviewer/`.
- Add `pyproject.toml` as the dependency, build, tool-configuration, and entry-point source of truth.
- Define the `arxiv-reviewer` console script.
- Preserve the top-level Python launcher only as a thin backward-compatible wrapper, or remove it if no longer useful after documentation is updated.
- Define a `dev` optional dependency group for pytest, coverage, Ruff, and build tooling.
- Support Python 3.11 through 3.13.
- Pin compatible dependency ranges in `pyproject.toml`.
- Commit an exact lock file produced from the tested environment.
- Ensure runtime imports do not initialize models or PDF libraries for `--help` and `status`.
- Finalize README installation, credential, and quick-start instructions.
- Add a license selected by the repository owner.
- Build both source and wheel distributions.

### Clean-install verification

Test from temporary environments with no repository-root import leakage:

1. Create a clean virtual environment.
2. Install the project from the repository.
3. Run `arxiv-reviewer --help`.
4. Run `arxiv-reviewer status` against a fixture database without credentials.
5. Install the development extra.
6. Run Ruff and the complete offline test suite.
7. Build the wheel and source distribution.
8. Create a second clean environment.
9. Install the built wheel rather than the working tree.
10. Repeat the CLI smoke tests from a directory outside the repository.

### Continuous integration

Add CI for Python 3.11 and 3.13 that:

- Installs `.[dev]`.
- Runs Ruff lint and formatting checks.
- Runs the offline tests with coverage.
- Builds the package.
- Installs the built wheel in a clean environment.
- Runs CLI smoke tests outside the checkout.
- Never requires provider credentials or network access beyond dependency installation.

### Final acceptance gate

- A fresh checkout can be installed using only documented commands.
- The wheel works outside the repository directory.
- `arxiv-reviewer --help` and `status` work without API credentials.
- Default tests make no network calls.
- CI passes on Python 3.11 and 3.13.
- The final README, example, benchmark data, and actual CLI behavior agree.

## 14. Required Test Matrix

| Area | Required cases |
| --- | --- |
| Configuration | Defaults, overrides, invalid limits, secret exclusion |
| Search | Query validation, normalization, deduplication, empty results, fatal failure |
| Relevance | Full success, partial failure, ranking, tie breaking, fallback, target limit |
| Parallelism | Concurrency one, concurrency three, deterministic reduction |
| Persistence | New thread, duplicate thread, unknown thread, resume, completed no-op, pending writes |
| PDF handling | Valid PDF, wrong type, corrupt, empty, oversized, timeout |
| Cache | Hit, miss, bypass, corrupt entry, atomic write |
| Token limits | Short, exact limit, oversized, section-aware sampling |
| Evidence | Valid, wrong page, missing excerpt, wrong paper, unsupported claim |
| Synthesis | Full success, invalid evidence reference, fallback |
| Reports | Complete, partial, no candidates, no selected papers, atomic output |
| Usage | Full metadata, partial metadata, unavailable metadata |
| CLI | All commands, exit codes, collisions, no-key status/help |
| Benchmark | Aggregation, deterministic metrics, secret redaction |
| Packaging | Editable/repository install, wheel install, outside-tree smoke test |

## 15. Implementation Rules

- Add or update tests in the same milestone as each behavior change.
- Prefer explicit dependencies over package-level wildcard exports.
- Keep network, model, cache, and filesystem effects behind injectable boundaries.
- Do not store secrets, PDF bytes, or full parsed papers in checkpoints or traces.
- Do not fabricate token usage, quality scores, speedups, or completion rates.
- Keep error messages useful but free of secrets and excessive provider payloads.
- Preserve deterministic ordering after every parallel reduction.
- Treat generated text as untrusted until its schema and evidence references validate.
- Do not rewrite Git history during artifact cleanup.
- Do not begin packaging until every preceding core milestone passes.

## 16. Completion Sequence

1. Establish baseline tests and dependency-injection seams.
2. Replace JSON snapshots with compact native SQLite persistence and true resume.
3. Implement abstract-first relevance evaluation and both parallel map-reduce stages.
4. Add page-grounded analysis, evidence validation, structured synthesis, and deterministic rendering.
5. Add retry/error handling, partial results, PDF safety, caches, token budgets, usage accounting, and graceful interruption.
6. Complete the offline suite, relevance evaluation, live benchmark tooling, verified example, documentation, and repository cleanup.
7. Only then migrate to final packaging, verify clean installation, add the console entry point, and enable package-focused CI.

No résumé bullet should be finalized until the implemented features and committed benchmark results support every claim.
