# Legacy Results

These are artifacts from the **pre-rewrite prototype**, kept only for reference.

- `reviews/` — Markdown literature reviews produced by the earlier pipeline.
- `checkpoints/` — serialized workflow state from that pipeline. These files use a
  state schema the current code no longer has (`current_paper_index`,
  `parsed_papers`, `chosen_papers`), and nothing reads them. The custom JSON
  checkpointer that wrote them has been replaced by LangGraph's SQLite checkpointer
  under `.arxiv-reviewer/`.
- `parsed/` — extracted paper text saved by hand while testing.

Nothing in this directory is used at runtime. It is slated for removal once a
verified example report exists under `examples/`.
