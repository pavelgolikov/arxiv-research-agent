"""Frozen evaluation datasets and metric runners.

Loads the repository `.env` on import so every eval entry point picks up model
credentials the same way the CLI does.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
