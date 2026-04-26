"""Use semantic chunking to split documents by topic shifts before retrieval.

Setup:
    bash setup/install_local_deps.sh
    export GROQ_API_KEY="your-groq-api-key"
"""

import os

from hayagriva import Hayagriva, HayagrivaConfig
from hayagriva.config import ChunkingConfig, RetrievalConfig


def _to_text(result) -> str:
    if isinstance(result, str):
        return result
    return "".join(result)


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY before running this example.")

    config = HayagrivaConfig(
        backend="groq",
        api_key=api_key,
        model="llama-3.1-8b-instant",
        chunking=ChunkingConfig(
            strategy="semantic",
            semantic_threshold=0.7,
        ),
        retrieval=RetrievalConfig(top_k=3),
    )

    rag = Hayagriva(config)
    rag.add_documents(
        [
            (
                "Hayagriva is connected with wisdom and recovered knowledge. "
                "The story emphasizes preserving sacred learning. "
                "Vector databases store embeddings for similarity search. "
                "Retrieval systems rank chunks by how closely they match a query."
            )
        ]
    )

    response = rag.ask(
        "Which retrieved idea is about preserving knowledge?",
        return_metadata=True,
    )

    print(response["answer"])
    print("\nChunking metadata:")
    print(response["chunking"])
    print("\nRetrieved chunks:")
    for chunk in response["chunks"]:
        print(f"- score={chunk['score']:.3f}: {chunk['text']}")


if __name__ == "__main__":
    main()
