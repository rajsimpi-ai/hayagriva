"""Retrieve chunks from FAISS without calling Groq for generation.

Setup:
    bash setup/install_local_deps.sh
"""

from hayagriva.config import ChunkingConfig, RetrievalConfig
from hayagriva.core.chunker import WordChunker
from hayagriva.core.embeddings import SentenceTransformerEmbeddings
from hayagriva.core.retriever import Retriever
from hayagriva.core.vectorstores import FaissVectorStore


def main() -> None:
    documents = [
        "Hayagriva is associated with recovered knowledge and wisdom.",
        "Chunking prepares text for embedding and retrieval.",
        "Generation uses retrieved context to produce grounded answers.",
    ]

    chunks, metadata = WordChunker(ChunkingConfig(chunk_size=10, overlap=2)).chunk(documents)
    embedder = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    store = FaissVectorStore()
    retriever = Retriever(embedder, store, RetrievalConfig(top_k=2))
    retriever.add(chunks, metadata=metadata)

    results = retriever.retrieve("What is Hayagriva associated with?")

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"{rank}. score={score:.3f} text={chunk}")


if __name__ == "__main__":
    main()
