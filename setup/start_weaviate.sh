#!/usr/bin/env bash
set -euo pipefail

# Starts a local Weaviate instance for:
# - examples/indexing/weaviate_index_chunks.py
# - examples/retrieval/weaviate_retrieve_chunks.py
# - examples/vectorstores/weaviate/vector_search.py
# - examples/vectorstores/weaviate/bm25_search.py
# - examples/vectorstores/weaviate/hybrid_search.py

docker compose -f setup/docker-compose.weaviate.yml up -d

echo "Weaviate is starting at http://localhost:8080"
echo "Use: export WEAVIATE_URL=http://localhost:8080"
