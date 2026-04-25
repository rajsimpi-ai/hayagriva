"""Use hierarchical chunking to keep parent context for retrieved child chunks.

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
            strategy="hierarchical",
            parent_chunk_size=120,
            chunk_size=35,
            overlap=8,
        ),
        retrieval=RetrievalConfig(top_k=3),
    )

    rag = Hayagriva(config)
    rag.add_documents(
        [
            (
                "Hayagriva is described as restoring lost knowledge and protecting wisdom. "
                "This larger parent passage gives broader context around recovery, learning, "
                "and preservation. RAG mirrors that pattern by retrieving small relevant child "
                "chunks while still keeping a connection to the larger source passage."
            )
        ]
    )

    result = rag.ask("Why is hierarchical chunking useful for RAG?")
    print(_to_text(result))


if __name__ == "__main__":
    main()
