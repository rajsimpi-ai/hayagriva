"""Load raw text values without chunking, embeddings, retrieval, or generation.

Setup:
    No external setup required.
"""

from hayagriva.ingestion.loaders import load_texts


def main() -> None:
    documents = load_texts(
        [
            "Hayagriva restores knowledge.",
            "   ",
            "RAG starts with clean source text.",
        ]
    )

    print(f"Loaded {len(documents)} documents:")
    for document in documents:
        print(f"- {document}")


if __name__ == "__main__":
    main()
