"""Retriever implementations."""
from __future__ import annotations

from typing import Any, Iterable, List, Tuple

from hayagriva.config import RetrievalConfig
from hayagriva.core.embeddings import SentenceTransformerEmbeddings
from hayagriva.utils.logger import get_logger
from hayagriva.utils.validator import validate_top_k

logger = get_logger(__name__)


class Retriever:
    """Coordinate query embedding and vector store search.

    Args:
        embedder: Embedding provider used to encode chunks and queries.
        vector_store: Vector store object implementing ``add`` and ``search``.
        config: Optional retrieval configuration.

    Attributes:
        embedder: Effective embedding provider.
        vector_store: Effective vector store backend.
        config: Effective retrieval configuration.
    """

    def __init__(
        self,
        embedder: SentenceTransformerEmbeddings,
        vector_store: Any,
        config: RetrievalConfig | None = None,
    ) -> None:
        """Initialize a retriever with an embedder and vector store.

        Args:
            embedder: Embedding provider used for chunks and queries.
            vector_store: Vector store backend implementing ``add`` and
                ``search``.
            config: Optional retrieval configuration.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()

    def add(self, chunks: Iterable[str], metadata: Iterable[dict] | None = None) -> None:
        """Embed chunks and add them to the configured vector store.

        Args:
            chunks: Iterable of chunk strings to index.
            metadata: Optional iterable of metadata dictionaries aligned with
                ``chunks``.
        """
        chunk_list = list(chunks)
        embeddings = self.embedder.embed(chunk_list)
        
        meta_list = list(metadata) if metadata else None
        self.vector_store.add(embeddings, chunk_list, metadata=meta_list)
        
        logger.info("Added %d chunks to vector store", len(chunk_list))

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """Retrieve chunks relevant to a query.

        Args:
            query: Natural-language search query.

        Returns:
            List of ``(chunk_text, score)`` pairs after applying the configured
            ``top_k`` and optional similarity threshold.

        Raises:
            ConfigurationError: If ``top_k`` is invalid.
            ValueError: If the vector store rejects the configured retrieval
                strategy.
        """
        top_k = validate_top_k(self.config.top_k)
        query_embedding = self.embedder.embed_query(query)
        
        results = self.vector_store.search(
            query_embedding, 
            top_k=top_k, 
            query_text=query,
            strategy=self.config.strategy,
            alpha=self.config.alpha
        )
        
        if self.config.similarity_threshold > 0:
            results = [r for r in results if r[1] >= self.config.similarity_threshold]
        logger.info("Retrieved %d chunks", len(results))
        return results
