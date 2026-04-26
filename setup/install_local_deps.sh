#!/usr/bin/env bash
set -euo pipefail

# Installs Hayagriva in editable mode with local CPU retrieval dependencies.
# Run this before FAISS-based examples and before Weaviate examples.

python3 -m pip install --upgrade pip
python3 -m pip install -e ".[cpu]"
