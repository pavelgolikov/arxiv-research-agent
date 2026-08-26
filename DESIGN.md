# Design notes

The README reports what the numbers are. This records how they were produced and why
they can be trusted — the methodology behind the benchmark, the experiments that
settled open questions, and the decisions that were measured and rejected.

Written for a reader who wants to interrogate the results rather than take them.

---

## 1. Building the retrieval benchmark

### Pooling

Judging every chunk against every question would be 5,700 decisions. Instead the
dataset uses **pooling**, the standard TREC approach: each of the four retrievers
contributes its top 10 for a question, and only the union is judged — 14 to 23 chunks
per question instead of the whole paper.

Labels are graded `0` irrelevant, `1` partial, `2` directly answers, because nDCG is a
graded metric and degenerates into something much blunter if labels are binary.

Thirty of the fifty questions are the exact strings `analysis.py` asks of every paper,
so the benchmark measures the real workload rather than a proxy for it. The other
twenty are paper-specific factual questions — named datasets, baselines, numbers —
that probe precise retrieval.

### Measuring pooling bias rather than assuming it away

Pooling has a known flaw. A chunk that no retriever returns is never judged, so it
counts as irrelevant by default and recall comes out higher than it should be. Rather
than assume the pools were deep enough, three chunks per question that **no** retriever
returned were sampled into the pool and judged blind alongside the rest.

They came back relevant more often than the 5% budgeted for, and how they split is the
useful part:

| Question type | Sampled unpooled chunks judged relevant |
| --- | --- |
| Facet — "what are the main findings?" | 14 / 90 = **15.6%** |
| Paper-specific — "which datasets were used?" | 0 / 60 = **0%** |
| Overall | 14 / 150 = 9.3% |

This is not a pool-depth problem, and deepening the pools would not fix it. Those
chunks were sampled from chunks *no retriever ranked at all*, whereas raising the depth
to 20 reaches ranks 11-20 — a different population. The actual cause is that broad
facet questions have diffuse relevance: "what are the main quantitative results?" is
genuinely answered by dozens of chunks spread through a 157-chunk paper, while a
precise question has two or three answers and the retrievers find them.

So the pools were kept as they are and the metrics were chosen to suit them.

### Which metrics the bias can and cannot corrupt

Pool depth is 10 and every retriever contributed its top 10, so at any cutoff k ≤ 10
**every chunk appearing in a ranked list has a label**. No unjudged chunk can enter a
result list and be silently scored as irrelevant. The numerator of every metric is
therefore exact, and only denominators are exposed:

| Metric | Exposure |
| --- | --- |
| MRR | None. Every ranked chunk is judged. |
| nDCG@10 | Ideal DCG only, on the 24 of 30 facet questions whose ideal top 10 is not already filled with top-grade chunks — and identically for all four retrievers, since they share one ground truth. The absolute level is slightly optimistic; the comparison between retrievers is not. |
| recall@5 | Directly. The size of the relevant set is exactly what the missed chunks corrupt. |

recall@5 was the weakest metric here even before this: with a mean of 7.7 relevant
chunks per facet question, five slots cannot hold them all, so it is capped at 71% on
facet questions however good the retriever is — 92% on paper-specific ones, where the
relevant sets are smaller.

The ablation therefore leads with **MRR and nDCG@10**, and reports **recall@5 split by
question type** rather than pooled into a single figure: clean on the paper-specific
half, flagged on the facet half.

`bpref` and `infAP`, estimators built for incomplete judgments, are the fully rigorous
alternative and are not implemented here.

### Keeping the guarantee true

"Every ranked chunk is judged" holds only while the index matches the one the pools
were built from. `evals/index/` is not committed, so `python -m evals.build_index`
rebuilds it from the committed chunk file — identical text, no PDF downloads — and
`--verify` replays all four retrievers over all fifty questions to confirm every
retrieved chunk was judged.

Re-embedding the entire corpus from scratch and re-checking still holds, so the
guarantee survives a clean clone. If drift ever breaks it, the check fails loudly
instead of quietly lowering recall. `evals/run_retrieval.py` enforces the same
invariant inline at scoring time.

<!-- eval:coverage -->
Verified at pool depth 10: 2,000 retrieved chunks checked across 50 questions, 0 unjudged.
<!-- /eval:coverage -->

---

## 2. Reading the ablation honestly

### Significance testing

The headline table alone would invite overclaiming: at fifty questions a gap of two or
three points of nDCG is indistinguishable from chance. Each difference is therefore
resampled with a paired bootstrap over the per-question scores, with a fixed seed so a
rerun reproduces the interval exactly.

<!-- eval:comparisons -->
| Comparison | Metric | Difference | 95% CI | Distinguishable |
| --- | --- | --- | --- | --- |
| `dense` → `bm25` | mrr | -0.108 | [-0.248, +0.027] | no |
| `dense` → `bm25` | ndcg@10 | -0.162 | [-0.262, -0.063] | **yes** |
| `dense` → `bm25` | recall@5 (specific) | -0.196 | [-0.329, -0.079] | **yes** |
| `dense` → `bm25` | recall@5 (facet) | -0.035 | [-0.123, +0.057] | no |
| `dense` → `hybrid` | mrr | +0.015 | [-0.079, +0.103] | no |
| `dense` → `hybrid` | ndcg@10 | +0.021 | [-0.039, +0.080] | no |
| `dense` → `hybrid` | recall@5 (specific) | -0.098 | [-0.194, -0.013] | **yes** |
| `dense` → `hybrid` | recall@5 (facet) | +0.071 | [+0.003, +0.141] | **yes** |
| `dense` → `hybrid-rerank` | mrr | +0.035 | [-0.058, +0.129] | no |
| `dense` → `hybrid-rerank` | ndcg@10 | +0.038 | [-0.041, +0.118] | no |
| `dense` → `hybrid-rerank` | recall@5 (specific) | -0.013 | [-0.150, +0.115] | no |
| `dense` → `hybrid-rerank` | recall@5 (facet) | +0.092 | [+0.006, +0.180] | **yes** |
| `hybrid` → `hybrid-rerank` | mrr | +0.021 | [-0.079, +0.122] | no |
| `hybrid` → `hybrid-rerank` | ndcg@10 | +0.017 | [-0.038, +0.074] | no |
| `hybrid` → `hybrid-rerank` | recall@5 (specific) | +0.085 | [-0.042, +0.210] | no |
| `hybrid` → `hybrid-rerank` | recall@5 (facet) | +0.021 | [-0.049, +0.090] | no |
<!-- /eval:comparisons -->

Five of sixteen comparisons survive. What that actually supports:

- **BM25 alone is clearly worst.** It loses 0.162 nDCG@10 to dense, and the interval is
  nowhere near zero.
- **Hybrid retrieval trades one kind of question for another.** Against dense it gains
  facet recall (+0.071) and loses paper-specific recall (−0.098). Fusing a keyword
  retriever in helps diffuse questions and hurts precise ones.
- **The cross-encoder reranker buys nothing this benchmark can distinguish.** Every
  `hybrid` → `hybrid-rerank` comparison spans zero, including the +0.017 nDCG@10 it
  appears to gain. It is the most expensive component in the stack — it pulls in `torch`
  and `sentence-transformers` — and fifty questions cannot show it earning that.
- **No MRR difference is distinguishable at all.** The first relevant chunk lands in
  much the same place whichever strategy is used.

A negative result on the flashiest component is still a result. Publishing it is more
credible than a suspiciously clean win.

### Why multi-query is excluded rather than scored

`--multi-query` expands each question into model-generated paraphrases, which retrieve
chunks the frozen pools never contained. The coverage guard would fire, and if it were
disabled the recall figures would be understated for reasons that have nothing to do
with the technique. Scoring it would need a re-pooled, re-labeled dataset.

---

## 3. Screening and search quality

### Choosing the relevance threshold

`RELEVANCE_THRESHOLD` was originally 4 because someone picked 4. It is now chosen by
sweeping the value through the real `select_papers_node` — not a copy of the selection
rule, which could drift from the production one — and scoring each result against the
labels.

The sweep chose **3**. Thresholds 3 and 4 give identical precision, but 3 recovers more
central papers (0.683 against 0.611) and under-fills one fewer query. Thresholds 2 and
3 select identically on this data, so the stricter of the two is taken: equal
performance on seven queries is not evidence that the looser value is equally safe on
unseen ones.

Two caveats the pooled numbers hide, which is why per-query figures are in the JSON.
The `interpretability` and `efficient_inference` queries came back
twelve-central-out-of-twelve, so their precision saturates at 1.0 no matter what is
selected. And central recall is ceiling-bound: with twelve central papers and a target
of four, no selection can exceed 0.33.

### The real bottleneck is arXiv search, not screening

A live run made this concrete. Asked for methods in language model self-improvement,
arXiv returned one on-topic paper and nine about federated LoRA, topic modeling, robot
platforms, and Byzantine-resilient SGD. The screener scored them 1 and 2 and was right
to. Selection quality is capped by what `search_arxiv` returns. Two follow-up studies
settled what to do about it.

### Rejected — an arXiv category filter

Measured in `evals/measure_categories.py`, results in `evals/results/categories.json`.
Fetching the categories of all 84 frozen candidates — metadata only, no relabeling —
shows the papers labeled irrelevant living in the same categories as the central ones:

```
irrelevant  cs.CV 10, cs.CL 8, cs.LG 8, cs.AI 4, cs.CR 3
central     cs.LG 27, cs.CL 24, cs.AI 23, stat.ML 3, cs.CV 3
```

| Filter | Kept | Irrelevant dropped | Central dropped | Central density |
| --- | --- | --- | --- | --- |
| cs.CL (any) | 36 | 20 | **20** | 52% → 67% |
| cs.CL + cs.LG (any) | 59 | 13 | **6** | 52% → 64% |
| cs.CL + cs.LG + cs.AI (any) | 66 | 12 | **3** | 52% → 62% |
| cs.\* (any CS) | 77 | 7 | **0** | 52% → 57% |

"Federated Sketching LoRA" and "Byzantine-Resilient SGD" are genuinely `cs.LG` — they
are simply not about self-improvement. Off-topic *for a query* is not the same as
off-topic *by category*, and arXiv categories cannot separate them.

The only free filter buys 5 points of density. Every filter that buys more takes
central papers with it. The trade is backwards: a junk candidate costs one
abstract-only model call and no download, while a lost central paper is unrecoverable.

This measures exclusion of what a filterless search returned, not what a filtered
search would return instead — that would need re-searching and would void all 84 hand
labels. It is the cheaper gate, and it closed the question.

### Accepted — searching deeper

Measured in `evals/measure_depth.py`, results in `evals/results/search_depth.json`.
`search_node` divides the result budget across the planned queries, so three queries at
`--max-results 10` gave each query four slots, and one weak query burned a third of the
candidate pool.

Replaying one run's exact queries at ten slots each — holding the queries fixed so
depth is the only variable:

| Per-query depth | Candidates | Relevant | Density |
| --- | --- | --- | --- |
| 4 | 12 | 3 | 25% |
| 10 | 30 | **8** | 27% |

The density column is the finding. arXiv's relevance ranking is flat over the first ten
results — ranks 4-9 yielded 5 of 18 relevant against 3 of 12 for ranks 0-3 — so going
deeper does not scrape the barrel, it gets more of the same quality. Among the papers
depth recovered was *Self-Refine: Iterative Refinement with Self-Feedback* at rank 7.

`--max-results` now defaults to 30 for this reason.

---

## 4. Grounding

### What the groundedness numbers measure

Citations that reach a finished report are valid by construction, because
`validate_claim` already discarded the invalid ones. Measuring referential integrity on
the finished output would return 100% every time and demonstrate nothing.

What is measured instead is the **survival rate of what the model proposed**, recorded
in `GroundedAnalysis` as `dropped_claims`, `dropped_evidence`, and `dropped_unsupported`.
Runs are read back from the LangGraph checkpoint rather than the rendered Markdown,
because the report keeps only page links while the checkpoint keeps the chunk IDs.

Citations are rejected at two stages and the rates keep them apart. Referential integrity
is measured against everything the model proposed; support integrity against what
survived the deterministic checks. Pooling them would charge the referential stage with
the judge's rejections, and the two stages move for entirely different reasons — the
hyphenation bug below moved the first and could not have moved the second.

Independent re-validation re-runs the three deterministic checks against the run's own
index. It is expected to return 100%; its value is catching a future change that
weakened validation or an index that drifted from the run it belongs to. It does not
re-run the judge. Asking the same model the same question twice measures that model's
consistency, not the run's soundness, so the judge is checked against hand labels in a
separate runner instead. A run recorded before the judge existed reports its support
integrity as "not judged" rather than 100%: nothing was rejected because nothing was
asked.

### The hyphenation bug

The first groundedness run reported 48.6% citation integrity. That was not
hallucination.

PDF extraction preserves the hyphens a typesetter inserted at line breaks, so a chunk
reads `lead- ing` where the paper reads `leading`. A model quoting the passage
faithfully writes `leading`, and the validator rejected it. Across two sample papers
this discarded 63% of citations that were in fact verbatim — the metric was measuring
the PDF extractor, not the model.

The diagnostic mattered: of the rejections, **zero** were hallucinated chunk IDs. All
were "excerpt not verbatim", and the excerpts read like genuine paper text. The fold
has to apply to both sides, because the inverse case appears too — a chunk containing
`fine- tuning` normalizes to `finetuning` while the model writes `fine-tuning`.

`normalize` now folds unicode form, case, whitespace, and hyphenation. On the captured
sample this took acceptance from 37% to 97% while still rejecting fabricated text,
paraphrases, hallucinated chunk IDs, and chunks belonging to another paper. A hyphen
absent from the source is still rejected: a model can only turn `fine tuning` into
`fine-tuning` by altering what it was shown.

Before the fix, the pipeline was silently discarding about half of its own valid work.

### What manual verification found

Referential integrity and claim support are different questions. The deterministic
checks answer the first: does this quote exist, on this page, in this paper. The second —
does the quote establish the sentence built on it — is now put to a model on every run,
but it was answered by a reader first, and it had to be: the labels below are what the
judge is scored against, and they would be worthless if a model had written them.

**First attempt, and why it was wrong.** Twelve citations of the example report were
read by hand and 8 were confirmed. That sample was drawn deliberately — spread across
papers and facets, and seeded with two citations already flagged as weak — so it
oversampled known problems. Its interval also spanned [39%, 86%], wide enough to be
close to uninformative.

**Second attempt.** Forty of the 165 citations in the committed groundedness runs,
drawn uniformly at random with a fixed seed, graded `2` establishes the claim, `1`
supports it partly, `0` no support. The draw landed evenly without stratification: 2 to
6 citations from each of the 8 papers, 4 to 9 from each of the 6 facets.

<!-- eval:claim_support -->
| Measure | Rate | 95% CI |
| --- | --- | --- |
| Excerpt establishes the claim | **77.5%** (31 of 40) | [62%, 88%] |
| Excerpt supports it at least partly | **100.0%** (31 + 9 of 40) | [91%, 100%] |
| Excerpt does not support the claim | 0 of 40 | — |
<!-- /eval:claim_support -->

The targeted sample understated support by 11 points, which is what a deliberately
adversarial draw should be expected to do. **No citation in the random sample failed
outright.**

### Where claim support falls short

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

Five facets are essentially clean. `experimental_setup` accounts for two thirds of every
partial grade in the sample, and the claims show why — they are enumerations:

- "model classes including GPT-2, LLaMA, Gemma, Bloom, and Mistral"
- "Datasets include custom documents (Apollo 11 mission, ARPA...)"
- "uses Llama-2, Llama-3, and Mistral as base models"
- "Baselines include full cache..."

The facet asks *"What datasets, baselines, and experimental setup are used?"*, whose
true answer is a list scattered through a paper. The model aggregates the list into one
claim and cites one fragment of it, so the excerpt supports part of the enumeration
rather than all of it. This is the diffuse-relevance problem from section 1 reappearing
at the claim layer rather than the retrieval layer.

**Excerpt truncation is not the cause.** Only 1 of the 9 partial grades sits at the
300-character `EVIDENCE_EXCERPT_CHARS` cap, and partial excerpts average slightly longer
than fully supporting ones (198 against 183 characters). Raising the limit would move
one item. A fix would have to act on the claim rather than the quote — requiring one
citation per enumerated item, or prompting `experimental_setup` for narrower claims.
Neither has been tried.

### Scoring the support judge

`analyze_facet` now grades every citation on the same rubric before the report is
written, and discards the `0`s. A model is marking another model's work, so the only
thing that makes the result worth anything is showing that it marks the way a person
does. `evals/labels/claim_support_labels.json` stores each hand grade next to its claim
and excerpt, and `run_claim_judge` replays those citations through the judge to compare.

**Forty positives cannot score a judge.** None of the uniform sample is a `0`, so a
judge that answers "supported" to everything scores 100% against it. Its catch rate is
not low on that set; it is undefined. `run_claim_judge` refuses to write a result while
that is true, rather than publishing an agreement figure that a constant function would
match.

**Where the negatives come from.** Thirty further citations from the same runs: ten more
real ones, and twenty whose excerpt was replaced by a different quote from the same
paper. Same paper on purpose. A quote lifted from an unrelated paper is a negative no
pipeline would ever produce — the deterministic checks reject it before the judge sees
it — whereas right-paper-wrong-sentence is exactly what those checks pass through, and
it is what the sheet has always told the labeler to grade `0`.

**They are labeled, not assumed.** A swapped quote can still support the claim by
coincidence; scoring the judge for "missing" one would be scoring it against a mistake.
So the swaps go through the same sheet, mixed unmarked among the real citations with the
key held in a separate file, and are graded by reading like everything else.

<!-- eval:claim_judge -->
<!-- /eval:claim_judge -->

Two rates come out, measuring opposite mistakes. The **catch rate** — of the citations a
person rejected, how many the judge rejected — is what the check buys. The **false-drop
rate** — of the citations a person accepted, how many the judge threw out — is what it
costs, in correct work deleted from the report. A judge that rejects everything scores a
perfect catch rate, so a single accuracy figure would let one hide behind the other.

**What this does not establish.** The constructed swaps are the easy end of the failure
mode. The failure that actually worries a reader is subtler — a quote that supports most
of a claim, or supports it without the qualifier the claim attaches — and nothing here
shows the judge catches those. The `experimental_setup` enumerations above are that
failure in its mildest form, and they are graded `1`, which the shipped threshold keeps.

**Why the threshold keeps partials.** Dropping grade `1` would discard 9 of 40 citations
in the measured sample to fix what is a claim-phrasing problem rather than a citation
problem: the claim aggregates a list and the quote names part of it. `run_claim_judge`
reports the stricter rule's rates alongside the shipped one so the cost is visible rather
than argued about, the way the screening threshold sweep does.

One difference from production is worth stating plainly: the runner grades one citation
per call, while `analyze_facet` grades a whole facet in one call and can read its items
against each other. Per-item is the leaner condition, but a large batching effect would
not show up in these numbers.

---

## Deliberately out of scope

LangSmith tracing, human-in-the-loop `interrupt`, `pyproject.toml` packaging, CI, and
any web UI or service.
