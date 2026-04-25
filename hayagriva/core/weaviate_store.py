"""Backward-compatible import path for the Weaviate vector store.

Prefer importing ``WeaviateVectorStore`` from ``hayagriva.core.vectorstores`` in
new code. This module remains so older imports continue to work.
"""

from hayagriva.core.vectorstores.weaviate import WeaviateVectorStore

__all__ = ["WeaviateVectorStore"]
