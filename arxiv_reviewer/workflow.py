"""LangGraph construction, SQLite persistence, and workflow invocation."""

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

from .analysis import (
    analyze_paper_node,
    screen_candidate_node,
    select_papers_node,
)
from .failures import is_retryable
from .rag import DEFAULT_DATA_DIR, DEFAULT_FETCH_K, DEFAULT_TOP_K
from .reporting import write_markdown_node
from .retrieval import search_node
from .review_types import (
    AnalysisOutcome,
    AnalyzeTask,
    EvidenceRef,
    GroundedAnalysis,
    PaperMetadata,
    RelevanceDecision,
    ReviewerState,
    ScreenOutcome,
    ScreenTask,
    SupportedClaim,
)

RECURSION_LIMIT = 150
DEFAULT_MAX_CONCURRENCY = 3
TERMINAL_STATUSES = frozenset({"complete", "partial", "empty"})

CHECKPOINT_TYPES = (
    AnalysisOutcome,
    EvidenceRef,
    GroundedAnalysis,
    PaperMetadata,
    RelevanceDecision,
    ScreenOutcome,
    SupportedClaim,
)

SEARCH_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    jitter=True,
    retry_on=is_retryable,
)


def fan_out_screening(state: ReviewerState) -> list[Send] | str:
    """Send every candidate to its own screening branch."""

    found_papers = state.get("found_papers", [])
    if not found_papers:
        return "write_markdown"

    return [
        Send(
            "screen_candidate",
            ScreenTask(
                paper=paper,
                search_position=position,
                user_query=state["user_query"],
            ),
        )
        for position, paper in enumerate(found_papers)
    ]


def fan_out_analysis(state: ReviewerState) -> list[Send] | str:
    """Send every selected paper to its own analysis branch."""

    selected_ids = state.get("selected_ids", [])
    if not selected_ids:
        return "write_markdown"

    positions = {
        paper.arxiv_id: position
        for position, paper in enumerate(state.get("found_papers", []))
    }
    papers = {paper.arxiv_id: paper for paper in state.get("found_papers", [])}

    return [
        Send(
            "analyze_paper",
            AnalyzeTask(
                paper=papers[arxiv_id],
                search_position=positions[arxiv_id],
                user_query=state["user_query"],
                thread_id=state["thread_id"],
                data_dir=state.get("data_dir", str(DEFAULT_DATA_DIR)),
                retriever_kind=state.get("retriever_kind", "hybrid-rerank"),
                top_k=state.get("top_k", DEFAULT_TOP_K),
                fetch_k=state.get("fetch_k", DEFAULT_FETCH_K),
                multi_query=state.get("multi_query", False),
            ),
        )
        for arxiv_id in selected_ids
    ]


def build_graph(checkpointer: SqliteSaver | None = None):
    """Assemble and compile the LangGraph workflow."""

    graph = StateGraph(ReviewerState)
    graph.add_node("search", search_node, retry_policy=SEARCH_RETRY_POLICY)
    graph.add_node("screen_candidate", screen_candidate_node, input_schema=ScreenTask)
    graph.add_node("select_papers", select_papers_node)
    graph.add_node("analyze_paper", analyze_paper_node, input_schema=AnalyzeTask)
    graph.add_node("write_markdown", write_markdown_node)

    graph.add_edge(START, "search")
    graph.add_conditional_edges(
        "search",
        fan_out_screening,
        ["screen_candidate", "write_markdown"],
    )
    graph.add_edge("screen_candidate", "select_papers")
    graph.add_conditional_edges(
        "select_papers",
        fan_out_analysis,
        ["analyze_paper", "write_markdown"],
    )
    graph.add_edge("analyze_paper", "write_markdown")
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


def thread_config(
    thread_id: str,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> dict[str, Any]:
    """Build the invocation config that binds work to one run thread."""

    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
        "max_concurrency": max_concurrency,
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
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
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
        "candidate_evaluations": [],
        "analysis_outcomes": [],
        "status": "running",
    }

    return persistent_graph(data_dir).invoke(
        state, config=thread_config(thread_id, max_concurrency)
    )


def resume_reviewer(
    thread_id: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> ReviewerState:
    """Continue an existing thread from its last recorded checkpoint."""

    graph = persistent_graph(data_dir)
    config = thread_config(thread_id, max_concurrency)
    snapshot = graph.get_state(config)

    if snapshot.created_at is None:
        raise KeyError(thread_id)
    if snapshot.values.get("status") in TERMINAL_STATUSES:
        return snapshot.values

    return graph.invoke(None, config=config)


def read_status(thread_id: str, data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    """Summarize a thread's recorded state without contacting any model."""

    snapshot = persistent_graph(data_dir).get_state(thread_config(thread_id))
    if snapshot.created_at is None:
        raise KeyError(thread_id)

    values = snapshot.values
    next_nodes = list(snapshot.next) or [task.name for task in snapshot.tasks]
    evaluations = values.get("candidate_evaluations", [])
    outcomes = values.get("analysis_outcomes", [])

    return {
        "thread_id": thread_id,
        "status": values.get("status", "running" if next_nodes else "unknown"),
        "next_nodes": next_nodes,
        "user_query": values.get("user_query", ""),
        "search_queries": values.get("search_queries", []),
        "candidate_papers": len(values.get("found_papers", [])),
        "screened_ok": sum(1 for item in evaluations if item.status == "ok"),
        "screened_failed": sum(1 for item in evaluations if item.status != "ok"),
        "selected_papers": len(values.get("selected_ids", [])),
        "analyzed_ok": sum(1 for item in outcomes if item.status == "ok"),
        "analyzed_failed": sum(1 for item in outcomes if item.status != "ok"),
        "indexed_chunks": sum(item.chunk_count for item in outcomes),
        "output": values.get("output", ""),
        "updated_at": snapshot.created_at,
    }
