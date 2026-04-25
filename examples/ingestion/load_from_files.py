"""Load .txt files from a file or directory without running RAG.

Setup:
    No external setup required.

Optional:
    export HAYAGRIVA_DOCS_PATH="./docs"
"""

import os
from pathlib import Path

from hayagriva.ingestion.loaders import load_from_paths


def main() -> None:
    docs_path = os.getenv("HAYAGRIVA_DOCS_PATH")
    if not docs_path:
        sample_dir = Path(__file__).parent / "sample_docs"
        sample_dir.mkdir(exist_ok=True)
        sample_file = sample_dir / "hayagriva.txt"
        sample_file.write_text(
            "Hayagriva represents restored knowledge.\n"
            "File ingestion reads .txt documents from paths.\n",
            encoding="utf-8",
        )
        docs_path = str(sample_dir)

    documents = load_from_paths([docs_path])

    print(f"Loaded {len(documents)} document(s) from {docs_path}:")
    for document in documents:
        print(document.strip())


if __name__ == "__main__":
    main()
