"""LangGraph construction, SQLite persistence, and workflow invocation."""

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .analysis import (
    advance_paper_node,
    extract_core_node,
    relevance_eval_node,
    route_after_advance_paper,
    route_after_extract_core,
    route_after_relevance_eval,
)
from .rag import DEFAULT_DATA_DIR, DEFAULT_FETCH_K, DEFAULT_TOP_K, ingest_node
from .reporting import write_markdown_node
from .retrieval import download_parse_node, route_after_search, search_node
from .review_types import (
    PaperAnalysis,
    PaperMetadata,
    ParsedPage,
    ParsedPaper,
    RelevanceDecision,
    ReviewerState,
)

RECURSION_LIMIT = 150

CHECKPOINT_TYPES = (
    PaperAnalysis,
    PaperMetadata,
    ParsedPage,
    ParsedPaper,
    RelevanceDecision,
)


def build_graph(checkpointer: SqliteSaver | None = None):
    """Assemble and compile the LangGraph workflow."""

    graph = StateGraph(ReviewerState)
    graph.add_node("search", search_node)
    graph.add_node("download_parse", download_parse_node)
    graph.add_node("ingest", ingest_node)
    graph.add_node("relevance_eval", relevance_eval_node)
    graph.add_node("advance_paper", advance_paper_node)
    graph.add_node("extract_core", extract_core_node)
    graph.add_node("write_markdown", write_markdown_node)

    graph.add_edge(START, "search")
    graph.add_conditional_edges(
        "search",
        route_after_search,
        {
            "download_parse": "download_parse",
            "write_markdown": "write_markdown",
        },
    )
    graph.add_edge("download_parse", "ingest")
    graph.add_edge("ingest", "relevance_eval")
    graph.add_conditional_edges(
        "relevance_eval",
        route_after_relevance_eval,
        {
            "extract_core": "extract_core",
            "advance_paper": "advance_paper",
        },
    )
    graph.add_conditional_edges(
        "extract_core",
        route_after_extract_core,
        {
            "advance_paper": "advance_paper",
            "write_markdown": "write_markdown",
        },
    )
    graph.add_conditional_edges(
        "advance_paper",
        route_after_advance_paper,
        {
            "download_parse": "download_parse",
            "write_markdown": "write_markdown",
        },
    )
    graph.add_edge("write_markdown", END)
    return graph.compile(checkpointer=checkpointer)


def checkpoint_db_path(data_dir: Path) -> Path:
    """Return the SQLite checkpoint database path."""

    return data_dir / "checkpoints.sqlite"


def checkpoint_serializer() -> JsonPlusSerializer:
    """Build a serializer that only reconstructs this project's own state types."""

    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


def open_checkpointer(data_dir: Path) -> SqliteSaver:
    """Open the SQLite checkpoint store, creating its schema when needed."""

    path = checkpoint_db_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(path), check_same_thread=False)
    checkpointer = SqliteSaver(connection, serde=checkpoint_serializer())
    checkpointer.setup()
    return checkpointer


def thread_config(thread_id: str) -> dict[str, Any]:
    """Build the invocation config that binds work to one run thread."""

    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }


def persistent_graph(data_dir: Path):
    """Compile the graph against the SQLite checkpoint store."""

    return build_graph(open_checkpointer(data_dir))


def thread_exists(data_dir: Path, thread_id: str) -> bool:
    """Report whether any checkpoint has been recorded for a thread."""

    snapshot = persistent_graph(data_dir).get_state(thread_config(thread_id))
    return snapshot.created_at is not None


def run_reviewer(
    user_query: str,
    max_results: int,
    target_papers: int,
    output: Path,
    thread_id: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    retriever_kind: str = "hybrid-rerank",
    top_k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    multi_query: bool = False,
) -> ReviewerState:
    """Invoke the compiled graph with initial user settings."""

    state: ReviewerState = {
        "user_query": user_query,
        "max_results": max_results,
        "target_papers": target_papers,
        "output": str(output),
        "thread_id": thread_id,
        "data_dir": str(data_dir),
        "retriever_kind": retriever_kind,
        "top_k": top_k,
        "fetch_k": fetch_k,
        "multi_query": multi_query,
        "current_paper_index": 0,
        "parsed_papers": {},
        "chunk_counts": {},
        "relevance_decisions": {},
        "chosen_papers": {},
        "status": "running",
    }

    return persistent_graph(data_dir).invoke(state, config=thread_config(thread_id))


def resume_reviewer(thread_id: str, data_dir: Path = DEFAULT_DATA_DIR) -> ReviewerState:
    """Continue an existing thread from its last recorded checkpoint."""

    graph = persistent_graph(data_dir)
    config = thread_config(thread_id)
    snapshot = graph.get_state(config)

    if snapshot.created_at is None:
        raise KeyError(thread_id)
    if not snapshot.next:
        return snapshot.values

    return graph.invoke(None, config=config)


def read_status(thread_id: str, data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    """Summarize a thread's recorded state without contacting any model."""

    snapshot = persistent_graph(data_dir).get_state(thread_config(thread_id))
    if snapshot.created_at is None:
        raise KeyError(thread_id)

    values = snapshot.values
    next_nodes = list(snapshot.next)

    return {
        "thread_id": thread_id,
        "status": values.get("status", "running" if next_nodes else "unknown"),
        "next_nodes": next_nodes,
        "user_query": values.get("user_query", ""),
        "search_queries": values.get("search_queries", []),
        "candidate_papers": len(values.get("found_papers", [])),
        "parsed_papers": len(values.get("parsed_papers", {})),
        "indexed_chunks": sum(values.get("chunk_counts", {}).values()),
        "selected_papers": len(values.get("chosen_papers", {})),
        "output": values.get("output", ""),
        "updated_at": snapshot.created_at,
    }


def retriever_defaults() -> tuple[str, int]:
    """Return the default retriever kind and top-k."""

    return "dense", DEFAULT_TOP_K
