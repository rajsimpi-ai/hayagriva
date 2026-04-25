"""Retrieve chunks from Weaviate without calling Groq for generation.

Setup:
    bash setup/install_local_deps.sh
    bash setup/start_weaviate.sh

Optional:
    export WEAVIATE_URL="http://localhost:8080"
    export WEAVIATE_INDEX_NAME="HayagrivaDocs"
"""

import os

from hayagriva.config import ChunkingConfig, RetrievalConfig, WeaviateConfig
from hayagriva.core.chunker import WordChunker
from hayagriva.core.embeddings import SentenceTransformerEmbeddings
from hayagriva.core.retriever import Retriever
from hayagriva.core.vectorstores import WeaviateVectorStore


def main() -> None:
    documents = [
        "Weaviate can run dense vector retrieval over stored chunks.",
        "BM25 retrieval searches exact terms in text fields.",
        "Hybrid retrieval combines vector similarity and keyword matching.",
    ]

    chunks, metadata = WordChunker(ChunkingConfig(chunk_size=10, overlap=2)).chunk(documents)
    embedder = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    store = WeaviateVectorStore(
        WeaviateConfig(
            url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            api_key=os.getenv("WEAVIATE_API_KEY"),
            index_name=os.getenv("WEAVIATE_INDEX_NAME", "HayagrivaDocs"),
        )
    )
    retriever = Retriever(embedder, store, RetrievalConfig(strategy="hybrid", alpha=0.5, top_k=2))
    retriever.add(chunks, metadata=metadata)

    results = retriever.retrieve("How does hybrid retrieval work?")

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"{rank}. score={score:.3f} text={chunk}")


if __name__ == "__main__":
    main()
