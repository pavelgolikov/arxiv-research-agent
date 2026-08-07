#!/usr/bin/env python3
"""Command-line entry point for the arXiv literature reviewer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from arxiv_reviewer import *


def main() -> int:
    """Validate command-line arguments and run the reviewer."""

    parser = argparse.ArgumentParser(
        description="Run an arXiv literature reviewer and write a Markdown report."
    )
    parser.add_argument(
        "--user-query",
        required=True,
        help="Research question or topic to review.",
    )
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--target-papers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("review.md"))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    load_dotenv()

    if args.max_results < 1:
        parser.error("--max-results must be at least 1.")
    if args.target_papers < 1:
        parser.error("--target-papers must be at least 1.")
    if args.target_papers > args.max_results:
        parser.error("--target-papers cannot be greater than --max-results.")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        parser.error("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    run_reviewer(
        user_query=args.user_query,
        max_results=args.max_results,
        target_papers=args.target_papers,
        output=args.output,
        checkpoint=args.checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
