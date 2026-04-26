"""Parsing utilities for ingested documents."""
from __future__ import annotations

from typing import Iterable, List, Tuple


def to_records(documents: Iterable[str]) -> List[Tuple[str, dict]]:
    """Convert plain documents into text and metadata records.

    Args:
        documents: Iterable of raw document strings.

    Returns:
        List of ``(text, metadata)`` tuples. Metadata is currently an empty
        dictionary for each document.
    """

    return [(doc, {}) for doc in documents]
