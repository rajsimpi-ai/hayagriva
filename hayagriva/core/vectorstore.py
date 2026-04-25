"""Backward-compatible import path for FAISS vector store implementations.

Prefer importing ``FaissVectorStore`` from ``hayagriva.core.vectorstores`` in
new code. This module remains so older imports continue to work.
"""

from hayagriva.core.vectorstores.faiss import FaissVectorStore

__all__ = ["FaissVectorStore"]
