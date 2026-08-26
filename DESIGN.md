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
in `GroundedAnalysis` as `dropped_claims` and `dropped_evidence`. Runs are read back
from the LangGraph checkpoint rather than the rendered Markdown, because the report
keeps only page links while the checkpoint keeps the chunk IDs.

Independent re-validation re-runs the three deterministic checks against the run's own
index. It is expected to return 100%; its value is catching a future change that
weakened validation or an index that drifted from the run it belongs to.

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

Twelve citations of the example report were checked by hand against the pages they
cite. **Eight were confirmed and four were rejected.** None was a fabrication, none
pointed at the wrong paper, and none cited a chunk that does not exist. All four were
real quotes asked to carry more than they say, in three distinct ways:

- **Truncation cut the supporting half.** One excerpt stops mid-word at 300 characters
  (`EVIDENCE_EXCERPT_CHARS`), just before the clause that would have supported the
  claim's second half.
- **The referent lay outside the quote.** One says "the difference of these vectors";
  the claim calls them activation vectors, which is correct in the paper but not
  established by the quoted span alone.
- **The claim added a word the quote does not carry.** One excerpt supports "crucial for
  understanding models" while the claim says "understanding *and controlling*". Another
  quotes the paper's title rather than a passage.

This is the most useful number in the repository. A human-judged support rate of 8/12
sits well below 94.3% machine-checked referential integrity, and the gap is not a
defect — it is the precise boundary of what a deterministic check can prove. The two
numbers belong side by side.

Two of the four failures have mechanical causes that could be addressed: raising the
excerpt limit, and prompting for self-contained passages. Neither has been tried.

---

## Deliberately out of scope

LangSmith tracing, human-in-the-loop `interrupt`, `pyproject.toml` packaging, CI, and
any web UI or service.
