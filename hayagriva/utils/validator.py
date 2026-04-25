"""Validation helpers for user inputs."""
from typing import Iterable, List

from hayagriva.exceptions import ConfigurationError, IngestionError


def ensure_texts(docs: Iterable[str]) -> List[str]:
    """Validate and filter document text.

    Args:
        docs: Iterable of candidate document values.

    Returns:
        Non-empty strings from ``docs``.

    Raises:
        IngestionError: If ``docs`` is ``None`` or contains no non-empty
            strings.
    """

    if docs is None:
        raise IngestionError("No documents provided for ingestion.")
    texts = [doc for doc in docs if isinstance(doc, str) and doc.strip()]
    if not texts:
        raise IngestionError("Documents must be non-empty strings.")
    return texts


def validate_top_k(top_k: int) -> int:
    """Validate a retrieval ``top_k`` value.

    Args:
        top_k: Requested number of retrieval results.

    Returns:
        The validated ``top_k`` value.

    Raises:
        ConfigurationError: If ``top_k`` is less than or equal to zero.
    """
    if top_k <= 0:
        raise ConfigurationError("top_k must be greater than zero")
    return top_k
