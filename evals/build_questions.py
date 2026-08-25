"""Build the retrieval question set for the frozen corpus.

Two kinds of question are included:

* `facet` — the exact questions the pipeline asks of every paper, so the benchmark
  measures the real workload rather than a proxy for it.
* `specific` — factual questions generated from each paper's own text, which probe
  precise retrieval (numbers, dataset names, baselines) that generic facet
  questions do not reach.
"""

import json

from pydantic import BaseModel, Field

from arxiv_reviewer.analysis import FACET_QUESTIONS
from arxiv_reviewer.gemini_client import generate_structured

from .config import (
    CHUNKS_FILE,
    DATA_DIR,
    EVAL_QUERIES,
    PAPER_SPECIFIC_QUESTIONS,
    PAPERS_FILE,
    QUESTIONS_FILE,
)


class GeneratedQuestions(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=6)


def paper_text(arxiv_id: str, limit: int = 24) -> str:
    """Return the opening chunks of a paper as context for question generation."""

    chunks = []
    with CHUNKS_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["arxiv_id"] == arxiv_id:
                chunks.append(record["text"])
            if len(chunks) >= limit:
                break
    return "\n\n".join(chunks)


def generate_specific(paper: dict) -> list[str]:
    """Ask the model for factual questions answerable from this paper."""

    prompt = (
        f"Write exactly {PAPER_SPECIFIC_QUESTIONS} factual questions that this "
        "paper answers.\n"
        "Each question must be answerable from a specific passage of the paper, "
        "not from general knowledge.\n"
        "Prefer questions about named datasets, baselines, metrics, numeric "
        "results, or specific components of the method.\n"
        "Do not number the questions and do not mention the paper's title.\n\n"
        f"Title: {paper['title']}\n\n"
        f"Paper text:\n{paper_text(paper['arxiv_id'])}"
    )
    return generate_structured(prompt, GeneratedQuestions).questions


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    papers = json.loads(PAPERS_FILE.read_text(encoding="utf-8"))
    query_text = {entry["query_id"]: entry["text"] for entry in EVAL_QUERIES}

    questions = []
    for paper in papers:
        arxiv_id = paper["arxiv_id"]

        for facet, text in FACET_QUESTIONS:
            questions.append(
                {
                    "question_id": f"{arxiv_id}:{facet}",
                    "arxiv_id": arxiv_id,
                    "kind": "facet",
                    "facet": facet,
                    "text": text,
                }
            )
        questions.append(
            {
                "question_id": f"{arxiv_id}:relevance_to_query",
                "arxiv_id": arxiv_id,
                "kind": "facet",
                "facet": "relevance_to_query",
                "text": query_text[paper["query_id"]],
            }
        )

        for index, text in enumerate(generate_specific(paper)):
            questions.append(
                {
                    "question_id": f"{arxiv_id}:specific{index}",
                    "arxiv_id": arxiv_id,
                    "kind": "specific",
                    "facet": None,
                    "text": text.strip(),
                }
            )

        made = len([q for q in questions if q["arxiv_id"] == arxiv_id])
        print(f"  {arxiv_id:<14} {made} questions")

    QUESTIONS_FILE.write_text(
        json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {QUESTIONS_FILE} ({len(questions)} questions)")


if __name__ == "__main__":
    main()
