"""Use a custom sentence-transformers embedding model with Groq generation.

Setup:
    bash setup/install_local_deps.sh
    export GROQ_API_KEY="your-groq-api-key"

Optional:
    export HAYAGRIVA_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
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

    embedding_model = os.getenv(
        "HAYAGRIVA_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    rag = Hayagriva(
        HayagrivaConfig(
            backend="groq",
            api_key=api_key,
            model="llama-3.1-8b-instant",
            embedding_model=embedding_model,
        )
    )

    rag.add_documents(
        [
            "Embedding models convert text into vectors for semantic comparison.",
            "A better domain-matched embedding model can improve retrieval quality.",
            "Groq handles generation after Hayagriva builds a context-aware prompt.",
        ]
    )

    result = rag.ask("Why would someone change the embedding model?")
    print(_to_text(result))


if __name__ == "__main__":
    main()
