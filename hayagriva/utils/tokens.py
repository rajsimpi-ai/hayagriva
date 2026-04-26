"""Token and text utilities."""
from typing import Iterable, List


def count_words(text: str) -> int:
    """Count words in a string using whitespace tokenization.

    Args:
        text: Input text.

    Returns:
        Number of whitespace-delimited tokens.
    """

    return len(text.split())


def batch_words(texts: Iterable[str]) -> List[int]:
    """Return word counts for a collection of strings.

    Args:
        texts: Iterable of input strings.

    Returns:
        Word count for each input string.
    """

    return [count_words(text) for text in texts]
