"""RAG example that returns answer + retrieval metadata.

Setup:
    bash setup/install_local_deps.sh
    export GROQ_API_KEY="your-groq-api-key"
"""

import os
from pprint import pprint

from hayagriva import Hayagriva, HayagrivaConfig


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY before running this example.")

    rag = Hayagriva(
        HayagrivaConfig(
            backend="groq",
            api_key=api_key,
            model="llama-3.1-8b-instant",
        )
    )

    rag.add_documents(
        [
            "Hayagriva is a symbol of wisdom and recovered knowledge.",
            "In retrieval-augmented generation, retrieval reduces hallucinations.",
            "Top-k retrieval selects the most relevant chunks for prompting.",
        ]
    )

    response = rag.ask(
        "How does retrieval help answer quality in this setup?",
        return_metadata=True,
    )

    print("Answer:\n")
    print(response["answer"])
    print("\nRetrieval metadata:\n")
    pprint(response["retrieval"])
    print("\nTop chunk preview:\n")
    print(response["chunks"][0]["text"])


if __name__ == "__main__":
    main()
