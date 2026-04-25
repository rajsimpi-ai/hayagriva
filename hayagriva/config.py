"""Configuration utilities for Hayagriva."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model settings used by Hayagriva.

    Attributes:
        embedding_model: Sentence-Transformers model name used to embed
            documents, chunks, and queries.
    """

    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class ChunkingConfig:
    """Configuration shared by text chunking strategies.

    Attributes:
        chunk_size: Target chunk size. Word-based chunkers interpret this as a
            word count.
        overlap: Number of words to carry from one word chunk into the next.
        strategy: Chunking strategy name. Supported values are ``"word"``,
            ``"recursive"``, ``"semantic"``, and ``"hierarchical"``.
        separators: Ordered separators used by the recursive chunker.
        semantic_threshold: Minimum adjacent-sentence cosine similarity used by
            semantic chunking before starting a new chunk.
        parent_chunk_size: Parent chunk size used by hierarchical chunking.
    """

    chunk_size: int = 200
    overlap: int = 20
    strategy: str = "word"  # "word", "recursive", "semantic", "hierarchical"
    separators: list = field(default_factory=lambda: ["\n\n", "\n", " ", ""])
    semantic_threshold: float = 0.8
    parent_chunk_size: int = 1000


@dataclass
class RetrievalConfig:
    """Configuration for retriever search behavior.

    Attributes:
        top_k: Maximum number of chunks to return for a query.
        similarity_threshold: Minimum score required for a result to be kept.
        strategy: Retrieval strategy. FAISS supports ``"vector"``; Weaviate
            supports ``"vector"``, ``"bm25"``, and ``"hybrid"``.
        alpha: Hybrid retrieval weight for Weaviate. ``0.0`` emphasizes sparse
            BM25 matching and ``1.0`` emphasizes dense vector matching.
    """

    top_k: int = 4
    similarity_threshold: float = 0.0
    strategy: str = "vector"  # "vector", "bm25", "hybrid"
    alpha: float = 0.5        # 0.0 = sparse (bm25), 1.0 = dense (vector)


@dataclass
class WeaviateConfig:
    """Connection settings for the Weaviate vector store.

    Attributes:
        url: Base URL for the Weaviate instance.
        api_key: Optional API key for authenticated hosted Weaviate instances.
        index_name: Weaviate class name used to store Hayagriva chunks.
    """
    
    url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    index_name: str = "HayagrivaDocs"


class HayagrivaConfig:
    """Top-level configuration for the Hayagriva facade.

    Args:
        backend: Generation backend. Currently only ``"groq"`` is supported.
        api_key: API key for the selected generation backend.
        model: Generation model name passed to the backend.
        embedding_model: Sentence-Transformers model used when ``models`` is
            not provided.
        vector_store: Vector store backend. Supported values are ``"faiss"``
            and ``"weaviate"``.
        weaviate: Optional Weaviate-specific configuration.
        chunking: Optional chunking configuration.
        models: Optional model configuration. When provided, this takes
            precedence over ``embedding_model``.
        retrieval: Optional retrieval configuration.

    Attributes:
        backend: Selected generation backend name.
        api_key: API key for the backend.
        model: Generation model name.
        vector_store: Selected vector store backend name.
        chunking: Chunking configuration instance.
        models: Model configuration instance.
        retrieval: Retrieval configuration instance.
        weaviate: Weaviate configuration instance.
    """

    def __init__(
        self,
        backend="groq",
        api_key=None,
        model="llama-3.1-8b-instant",
        embedding_model="all-MiniLM-L6-v2",
        vector_store="faiss",
        weaviate=None,
        chunking=None,
        models=None,
        retrieval=None,
    ):
        """Create a top-level Hayagriva configuration object.

        Args:
            backend: Generation backend name.
            api_key: API key for the selected generation backend.
            model: Generation model name.
            embedding_model: Embedding model name used when ``models`` is not
                provided.
            vector_store: Vector store backend name.
            weaviate: Optional Weaviate configuration.
            chunking: Optional chunking configuration.
            models: Optional model configuration.
            retrieval: Optional retrieval configuration.
        """
        self.backend = backend        # "groq"
        self.api_key = api_key        # API key for chosen backend
        self.model = model            # model name for backend
        self.vector_store = vector_store # "faiss" or "weaviate"

        # Existing config objects
        self.chunking = chunking or ChunkingConfig()
        self.models = models or ModelConfig(embedding_model=embedding_model)
        self.retrieval = retrieval or RetrievalConfig()
        self.weaviate = weaviate or WeaviateConfig()
