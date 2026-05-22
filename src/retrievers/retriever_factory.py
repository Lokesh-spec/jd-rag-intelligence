"""Factory for building and invoking JD retrievers over the Chroma index."""

from __future__ import annotations

import logging
from typing import Any

from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAI

from config.settings import (
    DEFAULT_TOP_K,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    LLM_TEMPERATURE,
    MMR_FETCH_K_MULTIPLIER,
    MMR_LAMBDA_MULT,
    SIMILARITY_SCORE_THRESHOLD,
)
from src.embeddings.embedding_pipeline import get_embedding_model
from src.retrievers.types import RetrieverType
from src.vectorstore.chroma_store import get_vector_store

logger = logging.getLogger(__name__)


class EmptyIndexError(RuntimeError):
    """Raised when the vector store has no documents to retrieve from."""


def _load_vector_store() -> Chroma:
    embeddings = get_embedding_model()
    vector_store = get_vector_store(embeddings)
    logger.debug("Connected to Chroma vector store")
    return vector_store


def fetch_all_documents(vector_store: Chroma) -> list[Document]:
    """Load all indexed JD chunks from Chroma as LangChain documents."""
    raw = vector_store.get()
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    if not documents:
        logger.warning("Vector store returned zero documents")
        return []

    if len(metadatas) != len(documents):
        logger.warning(
            "Metadata count (%d) does not match document count (%d); "
            "using empty metadata for missing entries",
            len(metadatas),
            len(documents),
        )
        metadatas = metadatas + [{}] * (len(documents) - len(metadatas))

    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(documents, metadatas)
    ]
    logger.info("Loaded %d document chunk(s) from index", len(docs))
    return docs


def build_dense_retriever(
    vector_store: Chroma,
    *,
    search_type: str,
    top_k: int,
    score_threshold: float | None = None,
) -> BaseRetriever:
    """Build a dense vector retriever (similarity or score threshold)."""
    search_kwargs: dict[str, Any] = {"k": top_k}
    if search_type == "similarity_score_threshold":
        if score_threshold is None:
            score_threshold = SIMILARITY_SCORE_THRESHOLD
        search_kwargs["score_threshold"] = score_threshold

    logger.debug(
        "Building dense retriever: search_type=%s, top_k=%d, kwargs=%s",
        search_type,
        top_k,
        search_kwargs,
    )
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def build_mmr_retriever(
    vector_store: Chroma,
    *,
    top_k: int,
    lambda_mult: float = MMR_LAMBDA_MULT,
    fetch_k: int | None = None,
) -> BaseRetriever:
    """Build an MMR retriever for diverse results."""
    if fetch_k is None:
        fetch_k = max(top_k * MMR_FETCH_K_MULTIPLIER, top_k)

    logger.debug(
        "Building MMR retriever: top_k=%d, fetch_k=%d, lambda_mult=%.2f",
        top_k,
        fetch_k,
        lambda_mult,
    )
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )


def build_bm25_retriever(
    documents: list[Document],
    *,
    top_k: int,
) -> BM25Retriever:
    """Build a BM25 retriever over the given document chunks."""
    if not documents:
        raise EmptyIndexError("Cannot build BM25 retriever: index has no documents")

    retriever = BM25Retriever.from_documents(documents=documents)
    retriever.k = top_k
    logger.debug("Built BM25 retriever with top_k=%d", top_k)
    return retriever


def build_hybrid_retriever(
    vector_store: Chroma,
    documents: list[Document],
    *,
    top_k: int,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    sparse_weight: float = HYBRID_SPARSE_WEIGHT,
) -> EnsembleRetriever:
    """Combine dense similarity and BM25 retrievers (notebook default: 0.8 / 0.2)."""
    dense = build_dense_retriever(
        vector_store,
        search_type="similarity",
        top_k=top_k,
    )
    sparse = build_bm25_retriever(documents, top_k=top_k)
    logger.debug(
        "Building hybrid retriever: dense_weight=%.2f, sparse_weight=%.2f",
        dense_weight,
        sparse_weight,
    )
    return EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[dense_weight, sparse_weight],
    )


def build_compressed_retriever(
    vector_store: Chroma,
    *,
    top_k: int,
    base_search_type: str = "similarity",
) -> ContextualCompressionRetriever:
    """Build contextual compression over a dense or MMR base retriever."""
    if base_search_type == "mmr":
        base = build_mmr_retriever(vector_store, top_k=top_k)
    else:
        base = build_dense_retriever(
            vector_store,
            search_type="similarity",
            top_k=top_k,
        )

    compressor = LLMChainExtractor.from_llm(
        OpenAI(temperature=LLM_TEMPERATURE)
    )
    logger.debug(
        "Building compressed retriever with base_search_type=%s", base_search_type
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base,
    )


def _invoke_retriever(retriever: BaseRetriever, query: str) -> list[Document]:
    results = retriever.invoke(query)
    logger.info("Retrieved %d document(s) for query", len(results))
    return results


def retrieve_documents(
    query: str,
    retriever_type: RetrieverType | str,
    top_k: int = DEFAULT_TOP_K,
    *,
    score_threshold: float | None = None,
    mmr_lambda_mult: float = MMR_LAMBDA_MULT,
    hybrid_dense_weight: float = HYBRID_DENSE_WEIGHT,
    hybrid_sparse_weight: float = HYBRID_SPARSE_WEIGHT,
) -> list[Document]:
    """
    Retrieve job-description chunks for a natural-language query.

    Args:
        query: Natural-language search query.
        retriever_type: Strategy name or RetrieverType enum value.
        top_k: Number of documents to return (where applicable).
        score_threshold: Override for similarity_score_threshold retrieval.
        mmr_lambda_mult: MMR diversity parameter (0 = max diversity, 1 = max relevance).
        hybrid_dense_weight: Weight for dense retriever in hybrid mode.
        hybrid_sparse_weight: Weight for BM25 retriever in hybrid mode.

    Returns:
        List of matching LangChain Document objects.

    Raises:
        ValueError: If retriever_type is not recognized.
        EmptyIndexError: If BM25/hybrid/compressed need documents but index is empty.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if isinstance(retriever_type, str):
        retriever_type = RetrieverType.from_string(retriever_type)

    logger.info(
        "Retrieving documents: type=%s, top_k=%d, query=%r",
        retriever_type.value,
        top_k,
        query[:80] + ("..." if len(query) > 80 else ""),
    )

    vector_store = _load_vector_store()

    if retriever_type == RetrieverType.SEMANTIC:
        retriever = build_dense_retriever(
            vector_store,
            search_type="similarity",
            top_k=top_k,
        )
        return _invoke_retriever(retriever, query)

    if retriever_type == RetrieverType.THRESHOLD:
        retriever = build_dense_retriever(
            vector_store,
            search_type="similarity_score_threshold",
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return _invoke_retriever(retriever, query)

    if retriever_type == RetrieverType.MMR:
        retriever = build_mmr_retriever(
            vector_store,
            top_k=top_k,
            lambda_mult=mmr_lambda_mult,
        )
        return _invoke_retriever(retriever, query)

    documents = fetch_all_documents(vector_store)

    if retriever_type == RetrieverType.BM25:
        if not documents:
            raise EmptyIndexError("No documents in index for BM25 retrieval")
        retriever = build_bm25_retriever(documents, top_k=top_k)
        return _invoke_retriever(retriever, query)

    if retriever_type == RetrieverType.HYBRID:
        if not documents:
            raise EmptyIndexError("No documents in index for hybrid retrieval")
        retriever = build_hybrid_retriever(
            vector_store,
            documents,
            top_k=top_k,
            dense_weight=hybrid_dense_weight,
            sparse_weight=hybrid_sparse_weight,
        )
        return _invoke_retriever(retriever, query)

    if retriever_type == RetrieverType.COMPRESSED:
        retriever = build_compressed_retriever(vector_store, top_k=top_k)
        return _invoke_retriever(retriever, query)

    raise ValueError(f"Unhandled retriever type: {retriever_type}")


# Backward-compatible aliases
retrieve_matching_jds = retrieve_documents


def dynamic_retriever(
    query: str,
    retriever_type: str,
    top_k: int = DEFAULT_TOP_K,
    **kwargs: Any,
) -> list[Document]:
    """Legacy entry point; prefer retrieve_documents()."""
    logger.debug("dynamic_retriever() is deprecated; use retrieve_documents()")
    return retrieve_documents(query, retriever_type, top_k, **kwargs)
