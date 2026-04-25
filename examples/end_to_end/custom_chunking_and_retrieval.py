"""RAG example with custom chunking and retrieval configuration.

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
            strategy="recursive",
            chunk_size=80,
            overlap=10,
        ),
        retrieval=RetrievalConfig(
            top_k=3,
            similarity_threshold=0.15,
        ),
    )

    rag = Hayagriva(config)
    rag.add_documents(
        [
            (
                "Hayagriva is described as restoring the Vedas after they were lost. "
                "In many interpretations, this restoration symbolizes preserving knowledge "
                "for future generations."
            ),
            (
                "RAG pipelines usually split documents into chunks and embed each chunk. "
                "At query time, retrieval returns the most relevant chunks and the LLM uses "
                "that context while generating a response."
            ),
        ]
    )

    result = rag.ask("Explain the connection between Hayagriva and retrieval in RAG.")
    print(_to_text(result))


if __name__ == "__main__":
    main()
