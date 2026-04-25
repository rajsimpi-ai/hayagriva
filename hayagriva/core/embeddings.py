"""Embedding providers."""
from __future__ import annotations

import importlib.util
from typing import Iterable, Optional

from hayagriva.exceptions import MissingDependencyError
from hayagriva.utils.logger import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbeddings:
    """Sentence-Transformers embedding wrapper.

    Args:
        model_name: Hugging Face or Sentence-Transformers model identifier.
        device: Optional device passed to ``SentenceTransformer`` such as
            ``"cpu"`` or ``"cuda"``.

    Attributes:
        model: Loaded ``SentenceTransformer`` model instance.
        model_name: Name of the embedding model in use.

    Raises:
        MissingDependencyError: If ``sentence-transformers`` is not installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> None:
        """Load a Sentence-Transformers model.

        Args:
            model_name: Model identifier to load.
            device: Optional execution device.

        Raises:
            MissingDependencyError: If ``sentence-transformers`` is missing.
        """
        if importlib.util.find_spec("sentence_transformers") is None:
            raise MissingDependencyError(
                "sentence-transformers is required for embedding. Install with `pip install sentence-transformers`."
            )
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: Iterable of text strings to encode.

        Returns:
            NumPy array with shape ``(len(texts), embedding_dimension)``.

        Raises:
            MissingDependencyError: If NumPy is not installed.
        """
        text_list = list(texts)
        logger.info("Encoding %d texts with %s", len(text_list), self.model_name)
        if importlib.util.find_spec("numpy") is None:
            raise MissingDependencyError("numpy is required for embedding operations.")
        import numpy as np

        return np.array(self.model.encode(text_list, convert_to_numpy=True))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            text: Query text to encode.

        Returns:
            One-dimensional NumPy vector for the query.

        Raises:
            MissingDependencyError: If NumPy is not installed.
        """
        if importlib.util.find_spec("numpy") is None:
            raise MissingDependencyError("numpy is required for embedding operations.")
        import numpy as np

        return np.array(self.model.encode([text], convert_to_numpy=True))[0]
