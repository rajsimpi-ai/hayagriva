"""Document loader utilities."""
from __future__ import annotations

from typing import Iterable, List

from hayagriva.exceptions import IngestionError
from hayagriva.utils.file_ops import read_text_files
from hayagriva.utils.validator import ensure_texts


def load_texts(docs: Iterable[str]) -> List[str]:
    """Validate and normalize raw document strings.

    Args:
        docs: Iterable of candidate document strings.

    Returns:
        Non-empty document strings.

    Raises:
        IngestionError: If no valid document strings are provided.
    """

    return ensure_texts(docs)


def load_from_paths(paths: Iterable[str]) -> List[str]:
    """Load UTF-8 text from files or directories.

    Args:
        paths: Iterable of file or directory paths. Directories are searched
            recursively for ``*.txt`` files.

    Returns:
        Text content read from all matching files.

    Raises:
        IngestionError: If a path cannot be read or does not exist.
    """

    try:
        return read_text_files(paths)
    except OSError as exc:  # pylint: disable=broad-except
        raise IngestionError(f"Failed to read paths: {exc}") from exc
