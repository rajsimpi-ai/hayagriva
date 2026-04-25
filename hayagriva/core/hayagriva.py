# hayagriva/core/hayagriva.py

from __future__ import annotations
from typing import Iterable, List, Optional

from hayagriva.config import HayagrivaConfig
from hayagriva.core.chunker import (
    WordChunker,
    RecursiveChunker,
    SemanticChunker,
    HierarchicalChunker
)
from hayagriva.core.context_builder import build_context
from hayagriva.core.embeddings import SentenceTransformerEmbeddings
from hayagriva.core.pipeline import build_prompt
from hayagriva.core.retriever import Retriever
from hayagriva.core.vectorstores import FaissVectorStore, WeaviateVectorStore
from hayagriva.utils.logger import get_logger
from hayagriva.utils.validator import ensure_texts

logger = get_logger(__name__)


class Hayagriva:
    """High-level interface for ingestion, retrieval, and generation.

    ``Hayagriva`` wires together the configured embedder, chunker, vector store,
    retriever, and Groq generator. Use lower-level classes directly when you
    only need part of the pipeline, such as chunking or indexing.

    Args:
        config: Optional top-level configuration. Defaults are Groq generation,
            FAISS vector storage, word chunking, and vector retrieval.

    Attributes:
        config: Effective Hayagriva configuration.
        embedder: Sentence-Transformers embedding provider.
        chunker: Chunker selected by ``config.chunking.strategy``.
        vector_store: Vector store selected by ``config.vector_store``.
        retriever: Retriever combining the embedder and vector store.
        generator: Groq generator used by ``ask``.

    Raises:
        ValueError: If the backend, vector store, or chunking strategy is not
            supported.
        MissingDependencyError: If a selected optional dependency is missing.
        GenerationError: If the Groq API key is missing.
    """

    def __init__(self, config: Optional[HayagrivaConfig] = None) -> None:
        """Initialize the configured RAG pipeline components.

        Args:
            config: Optional top-level configuration.

        Raises:
            ValueError: If an unsupported backend or vector store is selected.
            MissingDependencyError: If required optional dependencies are
                missing.
            GenerationError: If Groq configuration is invalid.
        """
        self.config = config or HayagrivaConfig()

        # Core components
        self.embedder = SentenceTransformerEmbeddings(
            self.config.models.embedding_model
        )

        # Initialize Chunker based on strategy
        strategy = self.config.chunking.strategy
        if strategy == "recursive":
            self.chunker = RecursiveChunker(self.config.chunking)
        elif strategy == "semantic":
            self.chunker = SemanticChunker(self.config.chunking, embedder=self.embedder)
        elif strategy == "hierarchical":
            self.chunker = HierarchicalChunker(self.config.chunking)
        else:
            self.chunker = WordChunker(self.config.chunking)
        
        if self.config.vector_store == "weaviate":
            self.vector_store = WeaviateVectorStore(self.config.weaviate)
        elif self.config.vector_store == "faiss":
            self.vector_store = FaissVectorStore()
        else:
            raise ValueError(f"Unknown vector store: {self.config.vector_store}")
            
        self.retriever = Retriever(
            self.embedder, self.vector_store, self.config.retrieval
        )

        # Backend selection
        if self.config.backend == "groq":
            from hayagriva.core.generators import GroqGenerator

            self.generator = GroqGenerator(
                api_key=self.config.api_key,
                model=self.config.model,
            )

        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

        self._documents: List[str] = []

    def add_documents(self, documents: Iterable[str]) -> None:
        """Ingest documents into the configured vector store.

        The method validates raw documents, chunks them with the configured
        chunker, embeds each chunk, and stores the results through the retriever.

        Args:
            documents: Iterable of non-empty document strings.

        Raises:
            IngestionError: If no valid document text is provided.
            MissingDependencyError: If embedding or vector-store dependencies
                are missing.
            ValueError: If vector-store inputs are inconsistent.
        """
        texts = ensure_texts(documents)
        self._documents.extend(texts)

        chunks, metadata = self.chunker.chunk(texts)
        logger.info("Chunked %d documents into %d chunks", len(texts), len(chunks))

        self.retriever.add(chunks, metadata)

    def ask(self, question: str, return_metadata: bool = False):
        """Answer a question using retrieval + generation.

        Args:
            question: Natural-language question to answer.
            return_metadata: When ``True``, return a structured dictionary with
                the answer plus retrieval, chunking, and model metadata.

        Returns:
            Generated answer text when ``return_metadata`` is ``False``. When
            ``return_metadata`` is ``True``, returns a dictionary with ``answer``,
            ``question``, ``chunks``, ``retrieval``, ``chunking``, and ``model``.

        Raises:
            GenerationError: If the configured generator fails.
            ConfigurationError: If retrieval configuration is invalid.
        """

        results = self.retriever.retrieve(question)
        contexts = [chunk for chunk, _ in results]

        context_block = build_context(contexts)
        prompt = build_prompt(question, contexts)

        logger.info("Built prompt with context length %d", len(context_block))

        result = self.generator.generate(prompt)

        if return_metadata:
            if hasattr(result, "__iter__") and not isinstance(result, str):
                answer = "".join(list(result))
            else:
                answer = result

            chunks = [
                {"rank": idx + 1, "text": chunk, "score": score}
                for idx, (chunk, score) in enumerate(results)
            ]

            return {
                "answer": answer,
                "question": question,
                "chunks": chunks,
                "retrieval": {
                    "strategy": self.config.retrieval.strategy,
                    "top_k": self.config.retrieval.top_k,
                    "similarity_threshold": self.config.retrieval.similarity_threshold,
                    "alpha": self.config.retrieval.alpha,
                },
                "chunking": {
                    "strategy": self.config.chunking.strategy,
                    "chunk_size": self.config.chunking.chunk_size,
                    "overlap": self.config.chunking.overlap,
                },
                "model": {
                    "backend": self.config.backend,
                    "model": self.config.model,
                    "embedding_model": self.config.models.embedding_model,
                    "vector_store": self.config.vector_store,
                },
            }

        return result

    def get_index_size(self) -> int:
        """Return the number of indexed chunks when the store exposes it.

        Returns:
            Number of chunks in stores that expose a ``chunks`` attribute, or
            ``0`` for stores where local size is not available.
        """
        if hasattr(self.vector_store, "chunks"):
            return len(self.vector_store.chunks)
        return 0

    @property
    def documents(self) -> List[str]:
        """Return a copy of raw documents added through ``add_documents``.

        Returns:
            List of original document strings retained by the facade.
        """
        return list(self._documents)
