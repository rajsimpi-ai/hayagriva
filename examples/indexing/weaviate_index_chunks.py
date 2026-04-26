"""Chunk documents, embed chunks, and store them in Weaviate without generation.

Setup:
    bash setup/install_local_deps.sh
    bash setup/start_weaviate.sh

Optional:
    export WEAVIATE_URL="http://localhost:8080"
    export WEAVIATE_INDEX_NAME="HayagrivaDocs"
"""

import os

from hayagriva.config import ChunkingConfig, WeaviateConfig
from hayagriva.core.chunker import WordChunker
from hayagriva.core.embeddings import SentenceTransformerEmbeddings
from hayagriva.core.vectorstores import WeaviateVectorStore


def main() -> None:
    documents = [
        "Weaviate stores vectors and text properties for retrieval.",
        "Hayagriva can provide embeddings and write them into Weaviate.",
    ]

    chunks, metadata = WordChunker(ChunkingConfig(chunk_size=10, overlap=2)).chunk(documents)
    embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2").embed(chunks)

    store = WeaviateVectorStore(
        WeaviateConfig(
            url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            api_key=os.getenv("WEAVIATE_API_KEY"),
            index_name=os.getenv("WEAVIATE_INDEX_NAME", "HayagrivaDocs"),
        )
    )
    store.add(embeddings, chunks, metadata=metadata)

    print(f"Indexed {len(chunks)} chunks in Weaviate")


if __name__ == "__main__":
    main()
