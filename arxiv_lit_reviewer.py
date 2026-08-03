#!/usr/bin/env python3
"""Command-line skeleton for a local arXiv literature reviewer.

This first implementation step only validates the requested run settings. It
does not call arXiv, Gemini, LangGraph, or PyMuPDF yet.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MAX_RESULTS = 10
DEFAULT_TARGET_PAPERS = 4
DEFAULT_OUTPUT = Path("review.md")


@dataclass(frozen=True)
class RunConfig:
    """Validated command-line settings for one literature review run."""

    query: str
    max_results: int
    target_papers: int
    output: Path
    checkpoint: Path


def positive_int(raw_value: str) -> int:
    """Parse a command-line value that must be a positive integer."""

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{raw_value!r} is not an integer."
        ) from exc

    if value < 1:
        raise argparse.ArgumentTypeError("Value must be at least 1.")

    return value


def default_checkpoint_path(output: Path) -> Path:
    """Return the checkpoint path that corresponds to a Markdown output path."""

    if output.suffix:
        return output.with_suffix(f"{output.suffix}.checkpoint.json")

    return output.with_name(f"{output.name}.checkpoint.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the single-script program."""

    parser = argparse.ArgumentParser(
        prog="arxiv_lit_reviewer.py",
        description=(
            "Prepare a local LangGraph run for an arXiv literature review."
        ),
    )
    parser.add_argument(
        "query",
        help="Research question or topic that the literature review should answer.",
    )
    parser.add_argument(
        "--max-results",
        type=positive_int,
        default=DEFAULT_MAX_RESULTS,
        help=f"Maximum number of arXiv results to inspect. Default: {DEFAULT_MAX_RESULTS}.",
    )
    parser.add_argument(
        "--target-papers",
        type=positive_int,
        default=DEFAULT_TARGET_PAPERS,
        help=(
            "Number of relevant papers to include in the review. "
            f"Default: {DEFAULT_TARGET_PAPERS}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown file to write in a later step. Default: {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "JSON checkpoint file to write in a later step. "
            "Default: the output path with .checkpoint.json appended."
        ),
    )
    return parser


def make_config(args: argparse.Namespace) -> RunConfig:
    """Convert parsed command-line arguments into a validated run config."""

    if args.target_papers > args.max_results:
        raise ValueError("--target-papers cannot be greater than --max-results.")

    checkpoint = args.checkpoint or default_checkpoint_path(args.output)
    return RunConfig(
        query=args.query,
        max_results=args.max_results,
        target_papers=args.target_papers,
        output=args.output,
        checkpoint=checkpoint,
    )


def require_gemini_api_key() -> None:
    """Exit with a clear error if Gemini credentials are not configured."""

    if os.environ.get("GEMINI_API_KEY"):
        return

    print(
        "GEMINI_API_KEY is not set. Set it before running the reviewer.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def print_config(config: RunConfig) -> None:
    """Print the validated run configuration for review."""

    print("ArXiv Literature Reviewer configuration")
    print(f"Query: {config.query}")
    print(f"Maximum arXiv results: {config.max_results}")
    print(f"Target relevant papers: {config.target_papers}")
    print(f"Output Markdown path: {config.output}")
    print(f"Checkpoint JSON path: {config.checkpoint}")
    print("Gemini API key: present")
    print()
    print(
        "Step 1 is complete. Search, PDF parsing, LangGraph execution, "
        "and Markdown writing are intentionally not implemented yet."
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, validate configuration, and print the run settings."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = make_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    require_gemini_api_key()
    print_config(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
