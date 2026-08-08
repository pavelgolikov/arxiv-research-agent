#!/usr/bin/env python3
"""Command-line entry point for the arXiv literature reviewer."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INVALID = 2

DEFAULT_DATA_DIR = Path(".arxiv-reviewer")
RETRIEVER_CHOICES = ["dense", "bm25", "hybrid", "hybrid-rerank"]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without importing runtime integrations."""

    parser = argparse.ArgumentParser(
        description="Run an arXiv literature reviewer and write a Markdown report."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Start a new review run.")
    run.add_argument("--query", "--user-query", dest="user_query", required=True)
    run.add_argument("--thread-id")
    run.add_argument("--max-results", type=int, default=10)
    run.add_argument("--target-papers", type=int, default=4)
    run.add_argument("--retriever", default="hybrid-rerank", choices=RETRIEVER_CHOICES)
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--fetch-k", type=int, default=20)
    run.add_argument("--multi-query", action="store_true")
    run.add_argument("--max-concurrency", type=int, default=3)
    run.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    run.add_argument("--output", type=Path, default=Path("review.md"))

    resume = commands.add_parser("resume", help="Continue an interrupted run.")
    resume.add_argument("--thread-id", required=True)
    resume.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    resume.add_argument("--max-concurrency", type=int, default=3)

    status = commands.add_parser("status", help="Report a run's recorded state.")
    status.add_argument("--thread-id", required=True)
    status.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    return parser


def require_api_key(parser: argparse.ArgumentParser) -> None:
    """Fail early when no model credentials are configured."""

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        parser.error("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")


def command_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Validate arguments and start a new review thread."""

    from arxiv_reviewer.workflow import run_reviewer, thread_exists

    if args.max_results < 1:
        parser.error("--max-results must be at least 1.")
    if args.target_papers < 1:
        parser.error("--target-papers must be at least 1.")
    if args.target_papers > args.max_results:
        parser.error("--target-papers cannot be greater than --max-results.")
    require_api_key(parser)

    thread_id = args.thread_id or str(uuid.uuid4())
    if args.thread_id and thread_exists(args.data_dir, thread_id):
        parser.error(
            f"Thread {thread_id} already exists. Use 'resume --thread-id {thread_id}'."
        )

    print(f"thread-id: {thread_id}")
    print(f"output:    {args.output}")

    run_reviewer(
        user_query=args.user_query,
        max_results=args.max_results,
        target_papers=args.target_papers,
        output=args.output,
        thread_id=thread_id,
        data_dir=args.data_dir,
        retriever_kind=args.retriever,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        multi_query=args.multi_query,
        max_concurrency=args.max_concurrency,
    )

    print(f"resume with: --thread-id {thread_id}")
    return EXIT_OK


def command_resume(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Continue an existing thread from its last checkpoint."""

    from arxiv_reviewer.workflow import resume_reviewer

    require_api_key(parser)

    try:
        resume_reviewer(
            args.thread_id,
            data_dir=args.data_dir,
            max_concurrency=args.max_concurrency,
        )
    except KeyError:
        print(f"Unknown thread: {args.thread_id}")
        return EXIT_INVALID

    print(f"thread-id: {args.thread_id}")
    return EXIT_OK


def command_status(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Print a thread's recorded state without contacting any model."""

    from arxiv_reviewer.workflow import read_status

    try:
        status = read_status(args.thread_id, data_dir=args.data_dir)
    except KeyError:
        print(f"Unknown thread: {args.thread_id}")
        return EXIT_INVALID

    for key, value in status.items():
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value) or "-"
        print(f"{key}: {value}")

    return EXIT_OK


def main() -> int:
    """Dispatch the requested command."""

    from dotenv import load_dotenv

    parser = build_parser()
    args = parser.parse_args()
    load_dotenv()

    handlers = {
        "run": command_run,
        "resume": command_resume,
        "status": command_status,
    }
    return handlers[args.command](args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
