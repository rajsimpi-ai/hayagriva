"""Use Weaviate as the vector store with dense vector retrieval.

Setup:
    bash setup/install_local_deps.sh
    bash setup/start_weaviate.sh
    export GROQ_API_KEY="your-groq-api-key"

Optional:
    export WEAVIATE_URL="http://localhost:8080"
    export WEAVIATE_INDEX_NAME="HayagrivaDocs"
"""

import os

from hayagriva import Hayagriva, HayagrivaConfig
from hayagriva.config import RetrievalConfig, WeaviateConfig


def _to_text(result) -> str:
    if isinstance(result, str):
        return result
    return "".join(result)


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY before running this example.")

    config = HayagrivaConfig(
        backend="groq",
        api_key=api_key,
        model="llama-3.1-8b-instant",
        vector_store="weaviate",
        weaviate=WeaviateConfig(
            url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            api_key=os.getenv("WEAVIATE_API_KEY"),
            index_name=os.getenv("WEAVIATE_INDEX_NAME", "HayagrivaDocs"),
        ),
        retrieval=RetrievalConfig(strategy="vector", top_k=3),
    )

    rag = Hayagriva(config)
    rag.add_documents(
        [
            "Weaviate can store externally generated vectors for semantic search.",
            "Dense vector retrieval compares embeddings to find related text.",
            "Hayagriva can use Groq for generation after Weaviate retrieval.",
        ]
    )

    result = rag.ask("What role does Weaviate play in this RAG setup?")
    print(_to_text(result))


if __name__ == "__main__":
    main()
