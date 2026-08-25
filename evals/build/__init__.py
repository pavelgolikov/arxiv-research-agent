"""Construction of the frozen evaluation datasets.

These modules produce the committed artifacts under `evals/data/` and the labeling
page that fills `evals/labels/`. They are run rarely and in a fixed order — see the
package docstring in `evals/__init__.py` — and are kept apart from the metric
runners, which only ever read what these write.
"""
