"""Pipeline definitions for RAG workflows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from hayagriva.core.context_builder import build_context


@dataclass
class RetrievalResult:
    """Container for retrieval output.

    Attributes:
        context: Context string assembled from retrieved chunks.
        sources: Source identifiers or labels associated with the context.
    """

    context: str
    sources: List[str]


def build_prompt(question: str, contexts: Iterable[str]) -> str:
    """Compose a RAG prompt from a question and retrieved contexts.

    Args:
        question: User question to answer.
        contexts: Retrieved context chunks to include in the prompt.

    Returns:
        Prompt string suitable for a chat-completion model.
    """

    context_block = build_context(contexts)
    return (
        "You are a helpful assistant. Use the provided context to answer the question.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    )
