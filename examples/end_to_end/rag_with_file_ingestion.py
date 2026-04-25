"""Load .txt files from paths, ingest them, and ask using Groq.

Setup:
    bash setup/install_local_deps.sh
    export GROQ_API_KEY="your-groq-api-key"

Optional:
    export HAYAGRIVA_DOCS_PATH="./docs"
"""

import os
from pathlib import Path

from hayagriva import Hayagriva, HayagrivaConfig
from hayagriva.ingestion.loaders import load_from_paths


def _to_text(result) -> str:
    if isinstance(result, str):
        return result
    return "".join(result)


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY before running this example.")

    docs_path = os.getenv("HAYAGRIVA_DOCS_PATH")
    if not docs_path:
        sample_dir = Path(__file__).parent / "sample_docs"
        sample_dir.mkdir(exist_ok=True)
        sample_file = sample_dir / "hayagriva.txt"
        sample_file.write_text(
            "Hayagriva represents the restoration and preservation of knowledge.\n"
            "RAG systems retrieve relevant text before generating answers.\n",
            encoding="utf-8",
        )
        docs_path = str(sample_dir)

    documents = load_from_paths([docs_path])

    rag = Hayagriva(
        HayagrivaConfig(
            backend="groq",
            api_key=api_key,
            model="llama-3.1-8b-instant",
        )
    )
    rag.add_documents(documents)

    result = rag.ask("What does the ingested document say about Hayagriva?")
    print(_to_text(result))


if __name__ == "__main__":
    main()
