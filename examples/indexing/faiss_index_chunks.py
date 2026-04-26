"""Chunk documents, embed chunks, and store them in FAISS without generation.

Setup:
    bash setup/install_local_deps.sh
"""

from hayagriva.config import ChunkingConfig
from hayagriva.core.chunker import WordChunker
from hayagriva.core.embeddings import SentenceTransformerEmbeddings
from hayagriva.core.vectorstores import FaissVectorStore


def main() -> None:
    documents = [
        "Hayagriva restores knowledge and protects wisdom.",
        "FAISS stores dense vectors for fast local similarity search.",
    ]

    chunks, metadata = WordChunker(ChunkingConfig(chunk_size=8, overlap=2)).chunk(documents)
    embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2").embed(chunks)

    store = FaissVectorStore()
    store.add(embeddings, chunks, metadata=metadata)

    print(f"Indexed {len(store.chunks)} chunks in FAISS")
    for chunk in store.chunks:
        print(f"- {chunk}")


if __name__ == "__main__":
    main()
