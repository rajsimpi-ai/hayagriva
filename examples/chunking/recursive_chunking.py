"""Split structured text with the recursive chunker.

Setup:
    No external setup required.
"""

from hayagriva.config import ChunkingConfig
from hayagriva.core.chunker import RecursiveChunker


def main() -> None:
    document = """
    Hayagriva

    Hayagriva is associated with wisdom and restored knowledge.

    RAG

    RAG systems retrieve relevant context before generating answers.
    Chunking controls how source text is prepared for embedding.
    """

    chunker = RecursiveChunker(ChunkingConfig(strategy="recursive", chunk_size=12))
    chunks, _ = chunker.chunk([document])

    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk.strip()}")


if __name__ == "__main__":
    main()
