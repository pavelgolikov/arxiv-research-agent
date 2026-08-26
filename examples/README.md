# Verified example

`example_review.md` is one complete run of the pipeline, kept so a reader can see
what the agent produces without running it.

## How it was produced

```bash
python arxiv_lit_reviewer.py run \
  --query "How is mechanistic interpretability used to understand transformer internals?" \
  --thread-id verified-example \
  --output examples/example_review.md
```

Run on 2026-08-25 with the shipping defaults: `--max-results 30`, `--target-papers 4`,
`--retriever hybrid-rerank`, `--top-k 5`, relevance threshold 3. Twenty-eight
candidates were screened from abstracts, four papers were selected and downloaded,
244 chunks were indexed, and 79 citations survived validation.

## What makes it worth looking at

Every statement in the review carries a page-anchored link, and every one of those
links was checked back to a real chunk of the paper it names. Twelve were then read by
hand against the pages they cite: 8 held up, 4 did not. `VERIFICATION.md` records which
checks were automated, which needed reading, and why the four failed — the reasons are
more interesting than the count.

The claims here are the model's, drawn only from retrieved excerpts of the cited
papers. Citation validation proves an excerpt exists where it is claimed to; it does
not prove the surrounding paper is correct, and arXiv papers are not peer reviewed.
