"""Ingest documents and ask a question using the CLI-like workflow."""
from hayagriva import Hayagriva


def main():
    docs = [
        "Hayagriva recovered the Vedas, symbolizing the retrieval of knowledge.",
        "RAG systems combine retrieval and generation for grounded answers.",
    ]
    rag = Hayagriva()
    rag.add_documents(docs)
    answer = rag.ask("What is Hayagriva known for?")
    print(answer)


if __name__ == "__main__":
    main()
