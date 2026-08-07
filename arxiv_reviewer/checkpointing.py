"""JSON checkpoint path handling and serialization."""

import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from .review_types import ReviewerState


def checkpoint_path(state: ReviewerState) -> Path:
    """Return the JSON checkpoint path for the current state."""

    if "checkpoint" in state:
        return state["checkpoint"]

    output = state.get("output", Path("review.md"))
    if output.suffix:
        return output.with_suffix(f"{output.suffix}.checkpoint.json")
    return output.with_name(f"{output.name}.checkpoint.json")


def to_jsonable(value: object) -> object:
    """Convert state values into JSON-compatible values."""

    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def save_checkpoint_node(state: ReviewerState) -> ReviewerState:
    """Write the current graph state to a JSON file."""

    path = checkpoint_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(state), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {"checkpoint": path}


def checkpointed_node(
    node_function: Callable[[ReviewerState], ReviewerState],
) -> Callable[[ReviewerState], ReviewerState]:
    """Wrap a graph node so its merged state is checkpointed afterward."""

    def wrapped_node(state: ReviewerState) -> ReviewerState:
        update = node_function(state)
        merged_state = dict(state)
        merged_state.update(update)
        save_checkpoint_node(merged_state)
        return update

    return wrapped_node
