"""LangGraph construction and workflow invocation."""

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from .analysis import (
    advance_paper_node,
    extract_core_node,
    relevance_eval_node,
    route_after_advance_paper,
    route_after_extract_core,
    route_after_relevance_eval,
)
from .checkpointing import checkpointed_node
from .review_types import ReviewerState
from .reporting import write_markdown_node
from .retrieval import download_parse_node, route_after_search, search_node


def build_graph():
    """Assemble and compile the LangGraph workflow."""

    graph = StateGraph(ReviewerState)
    graph.add_node("search", checkpointed_node(search_node))
    graph.add_node("download_parse", checkpointed_node(download_parse_node))
    graph.add_node("relevance_eval", checkpointed_node(relevance_eval_node))
    graph.add_node("advance_paper", checkpointed_node(advance_paper_node))
    graph.add_node("extract_core", checkpointed_node(extract_core_node))
    graph.add_node("write_markdown", checkpointed_node(write_markdown_node))

    graph.add_edge(START, "search")
    graph.add_conditional_edges(
        "search",
        route_after_search,
        {
            "download_parse": "download_parse",
            "write_markdown": "write_markdown",
        },
    )
    graph.add_edge("download_parse", "relevance_eval")
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
    return graph.compile()


def run_reviewer(
    user_query: str,
    max_results: int,
    target_papers: int,
    output: Path,
    checkpoint: Path | None,
) -> ReviewerState:
    """Invoke the compiled graph with initial user settings."""

    state: ReviewerState = {
        "user_query": user_query,
        "max_results": max_results,
        "target_papers": target_papers,
        "output": output,
        "current_paper_index": 0,
        "parsed_papers": {},
        "relevance_decisions": {},
        "chosen_papers": {},
    }
    if checkpoint:
        state["checkpoint"] = checkpoint

    return build_graph().invoke(state)
