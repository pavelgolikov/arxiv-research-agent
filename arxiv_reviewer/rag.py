"""Chunking, embedding, and retrieval over parsed papers."""

import os
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .review_types import ParsedPage, ReviewerState

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_DATA_DIR = Path(".arxiv-reviewer")
DEFAULT_TOP_K = 5
RETRIEVER_KINDS = ("dense",)

_UNSAFE_COLLECTION_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def data_dir_path(state: ReviewerState) -> Path:
    """Return the working-data directory recorded in graph state."""

    return Path(state.get("data_dir", str(DEFAULT_DATA_DIR)))


def chroma_dir(data_dir: Path | None = None) -> Path:
    """Return the persistent Chroma directory."""

    return (data_dir or DEFAULT_DATA_DIR) / "chroma"


def collection_name(thread_id: str) -> str:
    """Derive a Chroma-safe collection name for one run thread."""

    return f"run-{_UNSAFE_COLLECTION_CHARS.sub('-', thread_id)}"[:500]


def build_chunk_id(arxiv_id: str, page_number: int, index: int) -> str:
    """Build a stable identifier for one chunk of one page."""

    return f"{arxiv_id}:p{page_number}:c{index}"


def chunk_pages(arxiv_id: str, pages: list[ParsedPage]) -> list[Document]:
    """Split parsed pages into overlapping chunks that keep their page numbers."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents: list[Document] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue

        for index, piece in enumerate(splitter.split_text(text)):
            documents.append(
                Document(
                    page_content=piece,
                    metadata={
                        "arxiv_id": arxiv_id,
                        "page_number": page.page_number,
                        "chunk_id": build_chunk_id(arxiv_id, page.page_number, index),
                    },
                )
            )

    return documents


def get_embeddings() -> Embeddings:
    """Create the configured Gemini embedding model."""

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )


def open_store(
    thread_id: str,
    embeddings: Embeddings | None = None,
    data_dir: Path | None = None,
) -> Chroma:
    """Open the persistent vector store for one run thread."""

    directory = chroma_dir(data_dir)
    directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name(thread_id),
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(directory),
    )


def index_paper(
    thread_id: str,
    documents: list[Document],
    embeddings: Embeddings | None = None,
    data_dir: Path | None = None,
) -> int:
    """Embed and store one paper's chunks, replacing any existing copies."""

    if not documents:
        return 0

    store = open_store(thread_id, embeddings=embeddings, data_dir=data_dir)
    store.add_documents(
        documents,
        ids=[document.metadata["chunk_id"] for document in documents],
    )
    return len(documents)


def dense_retriever(
    thread_id: str,
    k: int = DEFAULT_TOP_K,
    embeddings: Embeddings | None = None,
    data_dir: Path | None = None,
) -> BaseRetriever:
    """Build a semantic-similarity retriever over the stored chunks."""

    store = open_store(thread_id, embeddings=embeddings, data_dir=data_dir)
    return store.as_retriever(search_kwargs={"k": k})


def get_retriever(
    thread_id: str,
    kind: str = "dense",
    k: int = DEFAULT_TOP_K,
    embeddings: Embeddings | None = None,
    data_dir: Path | None = None,
) -> BaseRetriever:
    """Build the requested retriever over the stored chunks."""

    if kind == "dense":
        return dense_retriever(
            thread_id, k=k, embeddings=embeddings, data_dir=data_dir
        )

    supported = ", ".join(RETRIEVER_KINDS)
    raise ValueError(f"Unsupported retriever kind {kind!r}. Supported: {supported}.")


def ingest_node(state: ReviewerState) -> ReviewerState:
    """Chunk the current paper and add it to the run's vector store."""

    current_paper_index = state.get("current_paper_index", 0)
    paper = state["found_papers"][current_paper_index]
    chunk_counts = dict(state.get("chunk_counts", {}))

    if paper.arxiv_id in chunk_counts:
        return {"chunk_counts": chunk_counts}

    parsed_paper = state["parsed_papers"][paper.arxiv_id]
    documents = chunk_pages(paper.arxiv_id, parsed_paper.pages)
    chunk_counts[paper.arxiv_id] = index_paper(
        state["thread_id"],
        documents,
        data_dir=data_dir_path(state),
    )

    return {"chunk_counts": chunk_counts}
