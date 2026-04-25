"""Context assembly utilities."""
from __future__ import annotations

from typing import Iterable, List


def build_context(chunks: Iterable[str]) -> str:
    """Concatenate retrieved chunks into a prompt-ready context block.

    Args:
        chunks: Iterable of chunk strings, usually ordered by retrieval rank.

    Returns:
        A single string with chunks separated by blank lines.
    """

    return "\n\n".join(chunks)
