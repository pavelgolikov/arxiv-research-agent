"""Shared configuration for the evaluation datasets."""

from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
DATA_DIR = EVALS_DIR / "data"
LABELS_DIR = EVALS_DIR / "labels"
RESULTS_DIR = EVALS_DIR / "results"
INDEX_DIR = EVALS_DIR / "index"

CANDIDATES_FILE = DATA_DIR / "screening_candidates.json"
PAPERS_FILE = DATA_DIR / "corpus_papers.json"
CHUNKS_FILE = DATA_DIR / "corpus_chunks.jsonl"
QUESTIONS_FILE = DATA_DIR / "retrieval_questions.json"
POOLS_FILE = DATA_DIR / "retrieval_pools.json"

SCREENING_LABELS = LABELS_DIR / "screening_labels.json"
RETRIEVAL_LABELS = LABELS_DIR / "retrieval_labels.json"

EVAL_THREAD_ID = "eval-corpus"

# Five research areas chosen to match the project's intended subject matter.
EVAL_QUERIES = [
    {
        "query_id": "agent_eval",
        "text": "evaluating LLM agent workflows and tool use",
    },
    {
        "query_id": "self_improvement",
        "text": "language model self-improvement and self-training",
    },
    {
        "query_id": "adversarial_reasoning",
        "text": "adversarial perturbations and robustness of LLM reasoning",
    },
    {
        "query_id": "interpretability",
        "text": "mechanistic interpretability of transformer internals",
    },
    {
        "query_id": "efficient_inference",
        "text": "efficient inference and KV cache optimization for LLMs",
    },
]

# One paper per query forms the retrieval corpus. These are chosen by hand from the
# frozen candidates above to be on-topic research papers with methods, experiments,
# and stated limitations, so every facet question has a real answer somewhere in the
# text. This is a property of the corpus, not of the retrievers being compared: the
# retrieval ablation measures finding the right chunk *within* a paper, and paper
# selection quality is what the separate screening dataset measures.
CORPUS_PAPERS = {
    "agent_eval": "2606.20023v2",
    "self_improvement": "2505.21444v2",
    "adversarial_reasoning": "2407.15549v3",
    "interpretability": "2511.09432v2",
    "efficient_inference": "2510.14973v2",
}

CANDIDATES_PER_QUERY = 12
POOL_DEPTH = 10
PAPER_SPECIFIC_QUESTIONS = 4
