"""Metadata structures for documents."""
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DocumentMetadata:
    """Metadata associated with an ingested document.

    Attributes:
        source: Optional source path, URL, or identifier for the document.
        attributes: Additional string metadata attached to the document.
    """

    source: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
