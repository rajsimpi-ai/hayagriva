"""Use Weaviate hybrid retrieval with configurable sparse/dense weighting.

Setup:
    bash setup/install_local_deps.sh
    bash setup/start_weaviate.sh
    export GROQ_API_KEY="your-groq-api-key"

Optional:
    export WEAVIATE_URL="http://localhost:8080"
    export WEAVIATE_INDEX_NAME="HayagrivaDocs"
    export HAYAGRIVA_HYBRID_ALPHA="0.5"
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

    alpha = float(os.getenv("HAYAGRIVA_HYBRID_ALPHA", "0.5"))

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
        retrieval=RetrievalConfig(strategy="hybrid", alpha=alpha, top_k=3),
    )

    rag = Hayagriva(config)
    rag.add_documents(
        [
            "Hybrid retrieval combines BM25 keyword matching with vector similarity.",
            "Alpha controls the balance between sparse and dense retrieval signals.",
            "A balanced hybrid setup can help when queries mix exact terms and concepts.",
        ]
    )

    result = rag.ask("How does alpha affect hybrid retrieval?")
    print(_to_text(result))


if __name__ == "__main__":
    main()
