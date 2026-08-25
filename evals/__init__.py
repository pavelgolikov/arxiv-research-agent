"""Frozen evaluation datasets and metric runners.

The package is split by direction of data flow. `build/` writes the datasets:
it searches arXiv, parses and chunks the corpus, generates questions, pools
candidates, and produces the labeling page. Everything else here only reads what
`build/` wrote, so a metric run can never quietly alter the ground truth it is
being scored against.

    evals/
      config.py      shared paths, queries, and corpus selection
      build_index.py materializes the committed chunks into a vector index
      build/         dataset construction (run rarely, in the order below)
      data/          frozen datasets, committed
      labels/        hand labels, committed
      results/       metric output, committed
      index/         vector index, rebuilt locally and not committed

Rebuild order, only if the corpus must change — each step consumes the previous
one's output, and re-running any of them invalidates the labels downstream:

    python -m evals.build.screening    # freeze arXiv candidate metadata
    python -m evals.build.corpus       # download, parse, chunk the corpus papers
    python -m evals.build.questions    # facet + paper-specific questions
    python -m evals.build.pools        # pooled candidates to judge
    python -m evals.build.label_tool   # regenerate the offline labeling page

Then label in the page, and run `python -m evals.build.verify_labels EXPORT
--write` to validate the export and write `labels/`.

Loads the repository `.env` on import so every eval entry point picks up model
credentials the same way the CLI does.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
