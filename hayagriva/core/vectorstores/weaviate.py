"""Weaviate vector store implementation."""

from __future__ import annotations

import importlib.util
import uuid
from typing import List, Sequence, Tuple

from hayagriva.config import WeaviateConfig
from hayagriva.exceptions import MissingDependencyError
from hayagriva.utils.logger import get_logger

logger = get_logger(__name__)


class WeaviateVectorStore:
    """Weaviate vector store wrapper."""

    def __init__(self, config: WeaviateConfig) -> None:
        if importlib.util.find_spec("weaviate") is None:
            raise MissingDependencyError(
                "weaviate-client is required for Weaviate support. Install with `pip install weaviate-client`."
            )
        import weaviate

        self.config = config
        self.client = weaviate.Client(
            url=config.url,
            auth_client_secret=(
                weaviate.auth.AuthApiKey(api_key=config.api_key) if config.api_key else None
            ),
        )

        self._ensure_schema()

    def _ensure_schema(self):
        """Ensure the class exists in Weaviate."""
        class_obj = {
            "class": self.config.index_name,
            "vectorizer": "none",
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                },
                {
                    "name": "parent_text",
                    "dataType": ["text"],
                },
            ],
        }

        if not self.client.schema.exists(self.config.index_name):
            self.client.schema.create_class(class_obj)
            logger.info("Created Weaviate class: %s", self.config.index_name)

    def add(self, embeddings, chunks: Sequence[str], metadata: Sequence[dict] | None = None) -> None:
        """Add embeddings and chunks to Weaviate."""
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks length mismatch")

        if len(chunks) == 0:
            return

        logger.info("Adding %d documents to Weaviate...", len(chunks))

        with self.client.batch as batch:
            batch.batch_size = 100
            for i, (vector, text) in enumerate(zip(embeddings, chunks)):
                properties = {
                    "text": text,
                }

                if metadata and i < len(metadata):
                    meta = metadata[i]
                    if "parent_text" in meta:
                        properties["parent_text"] = meta["parent_text"]

                batch.add_data_object(
                    data_object=properties,
                    class_name=self.config.index_name,
                    vector=vector,
                    uuid=uuid.uuid4(),
                )

        logger.info("Successfully added documents to Weaviate.")

    def search(
        self,
        query_embedding,
        top_k: int = 4,
        query_text: str = "",
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """Search for similar documents using vector, bm25, or hybrid search."""
        strategy = kwargs.get("strategy", "vector")
        alpha = kwargs.get("alpha", 0.5)

        query = self.client.query.get(self.config.index_name, ["text"])

        if strategy == "vector":
            vector = query_embedding.flatten().tolist()
            query = query.with_near_vector(
                {
                    "vector": vector,
                    "certainty": 0.0,
                }
            )
            additional_fields = ["certainty", "distance"]

        elif strategy == "bm25":
            if not query_text:
                logger.warning(
                    "BM25 search requested but no query text provided. Falling back to vector search."
                )
                return self.search(query_embedding, top_k, query_text, strategy="vector")

            query = query.with_bm25(query=query_text, properties=["text"])
            additional_fields = ["score"]

        elif strategy == "hybrid":
            if not query_text:
                logger.warning(
                    "Hybrid search requested but no query text provided. Falling back to vector search."
                )
                return self.search(query_embedding, top_k, query_text, strategy="vector")

            vector = query_embedding.flatten().tolist()
            query = query.with_hybrid(
                query=query_text,
                vector=vector,
                alpha=alpha,
                properties=["text"],
            )
            additional_fields = ["score"]

        else:
            raise ValueError(f"Unknown search strategy: {strategy}")

        response = query.with_limit(top_k).with_additional(additional_fields).do()

        if "errors" in response:
            logger.error("Weaviate search error: %s", response["errors"])
            return []

        results = []
        data = response.get("data", {}).get("Get", {}).get(self.config.index_name, [])

        for item in data:
            text = item.get("text")
            additional = item.get("_additional", {})

            if strategy == "vector":
                score = additional.get("certainty", 0.0)
            else:
                score = float(additional.get("score", 0.0))

            results.append((text, score))

        return results
