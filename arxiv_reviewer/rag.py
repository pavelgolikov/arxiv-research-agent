"""Chunking, embedding, and retrieval over parsed papers."""

import os
import re
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .review_types import ParsedPage, ReviewerState

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_DATA_DIR = Path(".arxiv-reviewer")
DEFAULT_TOP_K = 5
DEFAULT_FETCH_K = 20
HYBRID_WEIGHTS = (0.5, 0.5)
RETRIEVER_KINDS = ("dense", "bm25", "hybrid", "hybrid-rerank")

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


@lru_cache(maxsize=1)
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


def load_chunks(
    thread_id: str,
    arxiv_id: str | None = None,
    data_dir: Path | None = None,
) -> list[Document]:
    """Read stored chunks back out of the vector store in a stable order."""

    store = Chroma(
        collection_name=collection_name(thread_id),
        persist_directory=str(chroma_dir(data_dir)),
    )
    record = store.get(
        where={"arxiv_id": arxiv_id} if arxiv_id else None,
        include=["documents", "metadatas"],
    )

    documents = [
        Document(page_content=text, metadata=dict(metadata))
        for text, metadata in zip(record["documents"], record["metadatas"])
    ]
    documents.sort(key=lambda document: document.metadata["chunk_id"])
    return documents


class TopKRetriever(BaseRetriever):
    """Return only the highest-ranked documents from a wrapped retriever."""

    retriever: BaseRetriever
    k: int = DEFAULT_TOP_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.retriever.invoke(query)[: self.k]


def dense_retriever(
    thread_id: str,
    k: int = DEFAULT_TOP_K,
    arxiv_id: str | None = None,
    embeddings: Embeddings | None = None,
    data_dir: Path | None = None,
) -> BaseRetriever:
    """Build a semantic-similarity retriever over the stored chunks."""

    store = open_store(thread_id, embeddings=embeddings, data_dir=data_dir)
    search_kwargs: dict[str, object] = {"k": k}
    if arxiv_id:
        search_kwargs["filter"] = {"arxiv_id": arxiv_id}

    return store.as_retriever(search_kwargs=search_kwargs)


def bm25_retriever(
    thread_id: str,
    k: int = DEFAULT_TOP_K,
    arxiv_id: str | None = None,
    data_dir: Path | None = None,
) -> BaseRetriever:
    """Build a keyword-frequency retriever over the stored chunks."""

    from langchain_community.retrievers import BM25Retriever

    documents = load_chunks(thread_id, arxiv_id=arxiv_id, data_dir=data_dir)
    if not documents:
        scope = f"{thread_id!r}" if not arxiv_id else f"{thread_id!r}/{arxiv_id!r}"
        raise ValueError(f"No indexed chunks found for {scope}.")

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def hybrid_retriever(
    thread_id: str,
    k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    arxiv_id: str | None = None,
    embeddings: Embeddings | None = None,
    data_dir: Path | None = None,
) -> BaseRetriever:
    """Fuse semantic and keyword rankings with reciprocal rank fusion."""

    from langchain_classic.retrievers import EnsembleRetriever

    ensemble = EnsembleRetriever(
        retrievers=[
            dense_retriever(
                thread_id,
                k=fetch_k,
                arxiv_id=arxiv_id,
                embeddings=embeddings,
                data_dir=data_dir,
            ),
            bm25_retriever(
                thread_id, k=fetch_k, arxiv_id=arxiv_id, data_dir=data_dir
            ),
        ],
        weights=list(HYBRID_WEIGHTS),
    )
    return TopKRetriever(retriever=ensemble, k=k)


@lru_cache(maxsize=1)
def get_cross_encoder(model_name: str = RERANK_MODEL):
    """Load and cache the cross-encoder used for reranking."""

    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    return HuggingFaceCrossEncoder(model_name=model_name)


def apply_reranker(retriever: BaseRetriever, k: int = DEFAULT_TOP_K) -> BaseRetriever:
    """Rescore a retriever's candidates with a cross-encoder and keep the best."""

    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

    return ContextualCompressionRetriever(
        base_compressor=CrossEncoderReranker(model=get_cross_encoder(), top_n=k),
        base_retriever=retriever,
    )


def rerank_retriever(
    thread_id: str,
    k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    arxiv_id: str | None = None,
    embeddings: Embeddings | None = None,
    data_dir: Path | None = None,
) -> BaseRetriever:
    """Rescore fused candidates with a cross-encoder and keep the best."""

    base = hybrid_retriever(
        thread_id,
        k=fetch_k,
        fetch_k=fetch_k,
        arxiv_id=arxiv_id,
        embeddings=embeddings,
        data_dir=data_dir,
    )
    return apply_reranker(base, k=k)


def with_multi_query(retriever: BaseRetriever) -> BaseRetriever:
    """Expand each question into paraphrases before retrieving."""

    from langchain_classic.retrievers import MultiQueryRetriever

    from .gemini_client import gemini_llm

    return MultiQueryRetriever.from_llm(retriever=retriever, llm=gemini_llm())


def get_retriever(
    thread_id: str,
    kind: str = "dense",
    k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    arxiv_id: str | None = None,
    multi_query: bool = False,
    embeddings: Embeddings | None = None,
    data_dir: Path | None = None,
) -> BaseRetriever:
    """Build the requested retriever over the stored chunks."""

    if kind not in RETRIEVER_KINDS:
        supported = ", ".join(RETRIEVER_KINDS)
        raise ValueError(
            f"Unsupported retriever kind {kind!r}. Supported: {supported}."
        )

    reranked = kind == "hybrid-rerank"
    base_k = fetch_k if reranked else k

    if kind == "bm25":
        retriever = bm25_retriever(
            thread_id, k=base_k, arxiv_id=arxiv_id, data_dir=data_dir
        )
    elif kind == "dense":
        retriever = dense_retriever(
            thread_id,
            k=base_k,
            arxiv_id=arxiv_id,
            embeddings=embeddings,
            data_dir=data_dir,
        )
    else:
        retriever = hybrid_retriever(
            thread_id,
            k=base_k,
            fetch_k=fetch_k,
            arxiv_id=arxiv_id,
            embeddings=embeddings,
            data_dir=data_dir,
        )

    if multi_query:
        retriever = with_multi_query(retriever)
    if reranked:
        retriever = apply_reranker(retriever, k=k)

    return retriever


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
