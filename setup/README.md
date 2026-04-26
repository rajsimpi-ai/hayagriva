# Example Setup

Use these helper scripts before running examples that need local dependencies or external services.

## Common Local Setup

Most examples use:

- Groq for generation
- sentence-transformers for embeddings
- FAISS for local vector search

Run:

```bash
bash setup/install_local_deps.sh
export GROQ_API_KEY="your-groq-api-key"
```

This applies to FAISS, embedding, retrieval, and full local RAG examples:

- `examples/end_to_end/basic_rag_groq_faiss.py`
- `examples/end_to_end/rag_with_metadata.py`
- `examples/end_to_end/custom_chunking_and_retrieval.py`
- `examples/end_to_end/rag_with_file_ingestion.py`
- `examples/end_to_end/custom_embedding_model.py`
- `examples/chunking/semantic_chunking.py`
- `examples/chunking/hierarchical_chunking.py`
- `examples/indexing/embed_texts.py`
- `examples/indexing/faiss_index_chunks.py`
- `examples/retrieval/faiss_retrieve_chunks.py`

No external setup is required for:

- `examples/ingestion/load_texts.py`
- `examples/ingestion/load_from_files.py`
- `examples/chunking/word_chunking.py`
- `examples/chunking/recursive_chunking.py`

Optional example-specific environment variables:

```bash
export HAYAGRIVA_DOCS_PATH="./docs"
export HAYAGRIVA_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

## Weaviate Setup

The Weaviate examples need a running Weaviate instance in addition to the common local setup.

Run:

```bash
bash setup/install_local_deps.sh
bash setup/start_weaviate.sh
export GROQ_API_KEY="your-groq-api-key"
```

This applies to:

- `examples/indexing/weaviate_index_chunks.py`
- `examples/retrieval/weaviate_retrieve_chunks.py`
- `examples/vectorstores/weaviate/vector_search.py`
- `examples/vectorstores/weaviate/bm25_search.py`
- `examples/vectorstores/weaviate/hybrid_search.py`

Optional Weaviate environment variables:

```bash
export WEAVIATE_URL="http://localhost:8080"
export WEAVIATE_INDEX_NAME="HayagrivaDocs"
```

For hosted Weaviate:

```bash
export WEAVIATE_URL="https://your-cluster.weaviate.network"
export WEAVIATE_API_KEY="your-weaviate-api-key"
```

Stop the local Weaviate container with:

```bash
bash setup/stop_weaviate.sh
```
