"""Basic end-to-end RAG example using Groq and local FAISS retrieval.

Setup:
    bash setup/install_local_deps.sh
    export GROQ_API_KEY="your-groq-api-key"
"""

import os

from hayagriva import Hayagriva, HayagrivaConfig


def _to_text(result) -> str:
    if isinstance(result, str):
        return result
    return "".join(result)


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY before running this example.")

    rag = Hayagriva(
        HayagrivaConfig(
            backend="groq",
            api_key=api_key,
            model="llama-3.1-8b-instant",
            vector_store="faiss",
        )
    )

    rag.add_documents(
        [
            "Hayagriva is associated with wisdom and the recovery of sacred knowledge.",
            "Retrieval-augmented generation searches relevant documents before answering.",
            "FAISS is useful for local vector search during quick RAG prototyping.",
        ]
    )

    result = rag.ask("How does Hayagriva connect to the idea of retrieval?")
    print(_to_text(result))


if __name__ == "__main__":
    main()
