# Evaluation

What was measured, on what data, and what came out of it. The methodology behind these
numbers — how the datasets were built, which alternatives were tried and rejected, and
how far each result can be pushed — is in [`DESIGN.md`](DESIGN.md).

Every table on this page is generated from `evals/results/*.json` by
`python -m evals.render_tables --write`. No number here is typed by hand.

## Datasets

Two hand-labeled datasets under `evals/`, frozen and committed. Every table in this
section is generated from `evals/results/*.json` by `python -m evals.render_tables
--write`.

**Labeled datasets.** The ground truth every metric below is scored against, and the
amount of hand-labeling each one represents.

| Dataset | Size |
| --- | --- |
| Retrieval | 5 papers, 570 chunks, 50 questions, 312 judged-relevant chunks (246 fully answering, 66 partial) |
| Screening | 7 queries x 12 candidates, each labeled irrelevant / related / central |

Thirty of the fifty retrieval questions are the strings `analysis.py` asks of every
paper. The other twenty are paper-specific factual questions.

Methodology, experiments, and rejected alternatives: [`DESIGN.md`](DESIGN.md).

## Retrieval ablation

**Retrieval strategies compared.** Each row is one `--retriever` value run over the same
50 questions and scored against the frozen relevance labels. MRR is the reciprocal rank
of the first relevant chunk, so it catches a chunk that was retrieved but buried. nDCG@10
is graded and position-discounted. recall@5 is the share of relevant chunks reaching the
top 5 that the model is shown, split by question type because paper-specific and facet
questions behave differently. The final row is the highest recall@5 the judged pool
allows, so each column is read against its own ceiling rather than against 1.0.

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
intervals in [`DESIGN.md`](DESIGN.md#2-reading-the-ablation-honestly). `--multi-query` is
not scored: its paraphrases retrieve chunks outside the frozen pools.

## Screening quality

<!-- eval:screening -->
**Threshold sweep.** Each row is a candidate value of the relevance threshold in
`select_papers_node`, replayed against the screening labels. precision@4 is the
share of the four selected papers labeled central, then the share labeled central
or related. Central recall is the share of all central papers that were selected,
against the best any selection of four could reach. The last column counts queries
where fewer than four candidates cleared the threshold.

| Threshold | precision@4 (central) | precision@4 (related) | central recall | best achievable | queries under-filled |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.786 | 0.964 | 0.683 | _0.730_ | 0 |
| 2 | 0.810 | 1.000 | 0.683 | _0.730_ | 1 |
| 3 (current, **recommended**) | 0.810 | 1.000 | 0.683 | _0.730_ | 1 |
| 4 | 0.810 | 1.000 | 0.611 | _0.730_ | 2 |
| 5 | 0.786 | 0.857 | 0.540 | _0.730_ | 2 |

**Model score against label.** How many candidates the model gave each relevance
score, broken down by the label a person assigned to the same candidate. It shows
where the threshold can be placed without cutting into central papers.

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

## Groundedness

**Citation survival in a live run.** One run at the shipping defaults, read from the
LangGraph checkpoint rather than from the rendered report, which keeps only page links.
Each rate is what survived out of what the model proposed. Referential integrity is
measured against every proposed citation; support integrity against the citations that
passed the three deterministic checks and were then graded by the judge, so neither stage
is charged with the other's rejections. Re-validation re-runs the deterministic checks
against the run's own index and is expected to return 100%.

<!-- eval:groundedness -->
| Measure | Value |
| --- | --- |
| Papers analyzed | 4 across 1 run |
| Claim-support rate | **96.2%** (75 of 78 proposed claims kept) |
| Citation referential integrity | **96.2%** (77 of 80 proposed citations resolved) |
| Citation support integrity | **98.7%** (76 of 77 resolved citations judged to support their claim) |
| Citations per surviving claim | 1.01 |
| Independent re-validation | 100.0% (76 citations re-checked, 0 failures) |
<!-- /eval:groundedness -->

Every citation in that report was also read against its page by hand: 76 of 76 graded,
no rejections, 90.8% exact agreement with the support judge —
[`examples/VERIFICATION.md`](examples/VERIFICATION.md).

## Claim support

Validation runs in two layers: three exact checks prove the quote exists where it says
it does, then a model grades whether that quote supports the claim, and the failures are
discarded.

**Hand-graded citations.** Forty citations drawn at random from two runs and read one by
one against their claims by a person. Each row is the share of the 40 at that grade, with
a 95% confidence interval. The grades, claims, and excerpts are in
[`evals/labels/claim_support_labels.json`](evals/labels/claim_support_labels.json).

<!-- eval:claim_support -->
| Measure | Rate | 95% CI |
| --- | --- | --- |
| Excerpt establishes the claim | **77.5%** (31 of 40) | [62%, 88%] |
| Excerpt supports it at least partly | **100.0%** (31 + 9 of 40) | [91%, 100%] |
| Excerpt does not support the claim | 0 of 40 | — |
<!-- /eval:claim_support -->

**Partial grades by facet.** Of those 40, how many in each facet a reader graded `1`
rather than `2`, against the number of citations that facet contributed to the sample.

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

## Support judge accuracy

**Judge against reader.** The judge scored against 70 hand-graded citations: 50 the
pipeline produced, and 20 whose excerpt was swapped for a different quote from the same
paper so the set contains failures the deterministic checks cannot see. The first two
rows measure opposite mistakes — the catch rate is what the check is worth, the
false-drop rate is what it costs in correct work discarded — and neither means anything
without the other, since a judge that rejects everything scores a perfect catch rate.

<!-- eval:claim_judge -->
| Measure | Rate | 95% CI |
| --- | --- | --- |
| Rejects a citation a reader also rejected (catch rate) | **100.0%** (15 of 15) | [80%, 100%] |
| Rejects a citation a reader kept (false-drop rate) | **3.6%** (2 of 55) | [1%, 12%] |
| Exact grade agreement | **87.1%** (61 of 70) | [77%, 93%] |
<!-- /eval:claim_judge -->

Against the 50 real citations alone the judge dropped none. How the set was built, why
the swaps were graded rather than assumed, and where the false drops fell:
[`DESIGN.md`](DESIGN.md#scoring-the-support-judge).

Regenerate the sheets with `python -m evals.build.claim_support` and
`python -m evals.build.claim_support --judge-set`, then score the judge with
`python -m evals.run_claim_judge`.

## Reproducing

```bash
python -m evals.build_index --verify          # rebuild the index, check pool coverage
python -m evals.run_retrieval                 # ablation
python -m evals.run_screening                 # threshold sweep
python -m evals.run_groundedness --thread-id THREAD_ID
python -m evals.run_claim_judge               # judge against the hand labels
python -m evals.render_tables --write         # regenerate every table above
```
