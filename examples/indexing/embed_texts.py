"""Generate embeddings for text without storing them in a vector database.

Setup:
    bash setup/install_local_deps.sh
"""

from hayagriva.core.embeddings import SentenceTransformerEmbeddings


def main() -> None:
    embedder = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    texts = [
        "Hayagriva restores knowledge.",
        "RAG retrieves context before answering.",
    ]

    embeddings = embedder.embed(texts)

    print(f"Embedded {len(texts)} texts")
    print(f"Embedding matrix shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
