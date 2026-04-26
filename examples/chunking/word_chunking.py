"""Split documents with the default word-window chunker.

Setup:
    No external setup required.
"""

from hayagriva.config import ChunkingConfig
from hayagriva.core.chunker import WordChunker


def main() -> None:
    chunker = WordChunker(ChunkingConfig(chunk_size=8, overlap=2))
    chunks, metadata = chunker.chunk(
        [
            (
                "Hayagriva restores knowledge and RAG retrieves context before "
                "generation so answers remain grounded in source material."
            )
        ]
    )

    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk}")
    print(f"Metadata records: {metadata}")


if __name__ == "__main__":
    main()
