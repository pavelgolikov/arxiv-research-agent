# arXiv Literature Reviewer

A LangGraph workflow that searches arXiv, evaluates papers against a research
question, extracts structured notes, and writes a Markdown literature review
with Gemini through LangChain.

## Repository layout

- `arxiv_lit_reviewer.py` — backward-compatible command-line launcher.
- `arxiv_reviewer/` — application package, split by responsibility.
- `results/reviews/` — historical generated literature reviews.
- `results/checkpoints/` — saved workflow state from historical runs.
- `results/parsed/` — extracted paper text retained from development.
- `portfolio_upgrade_plan.md` — roadmap for the production-grade portfolio version.

## Run

Install the dependencies, configure `GEMINI_API_KEY` or `GOOGLE_API_KEY`, and run:

```bash
pip install -r requirements.txt
python arxiv_lit_reviewer.py \
  --user-query "What is the latest and greatest in model self-improvement?"
```

The existing CLI options and defaults are unchanged by the source-code
reorganization.
