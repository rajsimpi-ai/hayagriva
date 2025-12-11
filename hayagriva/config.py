"""Configuration utilities for Hayagriva."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for embedding and generation models."""

    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gpt-4o"
    vector_store: str = "faiss"
    openai_api_key: Optional[str] = None


@dataclass
class ChunkingConfig:
    """Configuration for the text chunker."""

    chunk_size: int = 200
    overlap: int = 20


@dataclass
class RetrievalConfig:
    """Configuration for retrieval behavior."""

    top_k: int = 4
    similarity_threshold: float = 0.0


@dataclass
class HayagrivaConfig:
    """Aggregate configuration for the Hayagriva orchestrator."""

    models: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
