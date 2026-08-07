"""Public API for the arXiv literature reviewer."""

from .analysis import (
    RELEVANCE_TEXT_CHARS,
    advance_paper_node,
    extract_core_node,
    relevance_eval_node,
    route_after_advance_paper,
    route_after_extract_core,
    route_after_relevance_eval,
)
from .checkpointing import (
    checkpoint_path,
    checkpointed_node,
    save_checkpoint_node,
    to_jsonable,
)
from .gemini_client import (
    GEMINI_MODEL,
    generate_structured,
    generate_text,
    gemini_llm,
)
from .review_types import (
    LiteratureReview,
    PaperAnalysis,
    PaperMetadata,
    ParsedPaper,
    RelevanceDecision,
    ReviewerState,
    SearchPlan,
)
from .reporting import render_markdown_fallback, write_markdown_node
from .retrieval import (
    download_parse_node,
    make_search_plan,
    route_after_search,
    search_arxiv,
    search_node,
)
from .workflow import build_graph, run_reviewer

__all__ = [
    "GEMINI_MODEL",
    "RELEVANCE_TEXT_CHARS",
    "LiteratureReview",
    "PaperAnalysis",
    "PaperMetadata",
    "ParsedPaper",
    "RelevanceDecision",
    "ReviewerState",
    "SearchPlan",
    "advance_paper_node",
    "build_graph",
    "checkpoint_path",
    "checkpointed_node",
    "download_parse_node",
    "extract_core_node",
    "gemini_llm",
    "generate_structured",
    "generate_text",
    "make_search_plan",
    "relevance_eval_node",
    "render_markdown_fallback",
    "route_after_advance_paper",
    "route_after_extract_core",
    "route_after_relevance_eval",
    "route_after_search",
    "run_reviewer",
    "save_checkpoint_node",
    "search_arxiv",
    "search_node",
    "to_jsonable",
    "write_markdown_node",
]
