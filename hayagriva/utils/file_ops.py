"""File utilities for reading and writing text documents."""
from pathlib import Path
from typing import Iterable, List

from hayagriva.exceptions import IngestionError


def read_text_files(paths: Iterable[str]) -> List[str]:
    """Read UTF-8 text from files and directories.

    Args:
        paths: Iterable of filesystem paths. File paths are read directly.
            Directory paths are searched recursively for ``*.txt`` files.

    Returns:
        Text content from all discovered files.

    Raises:
        IngestionError: If any path does not exist.
        OSError: If a file cannot be read.
    """

    contents: List[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise IngestionError(f"Path does not exist: {path}")
        if path.is_dir():
            for child in path.glob("**/*.txt"):
                contents.append(child.read_text(encoding="utf-8"))
        else:
            contents.append(path.read_text(encoding="utf-8"))
    return contents
