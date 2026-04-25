# Examples

The examples are organized by what part of the pipeline they demonstrate.

## Ingestion

Load raw strings or `.txt` files:

- `examples/ingestion/load_texts.py`
- `examples/ingestion/load_from_files.py`

## Chunking

Run chunkers directly:

- `examples/chunking/word_chunking.py`
- `examples/chunking/recursive_chunking.py`
- `examples/chunking/semantic_chunking.py`
- `examples/chunking/hierarchical_chunking.py`

## Indexing

Embed chunks and store them:

- `examples/indexing/embed_texts.py`
- `examples/indexing/faiss_index_chunks.py`
- `examples/indexing/weaviate_index_chunks.py`

## Retrieval

Retrieve chunks without generation:

- `examples/retrieval/faiss_retrieve_chunks.py`
- `examples/retrieval/weaviate_retrieve_chunks.py`

## End To End

Full RAG flows:

- `examples/end_to_end/basic_rag_groq_faiss.py`
- `examples/end_to_end/rag_with_metadata.py`
- `examples/end_to_end/rag_with_file_ingestion.py`
- `examples/end_to_end/custom_chunking_and_retrieval.py`
- `examples/end_to_end/custom_embedding_model.py`

## Weaviate

Weaviate-specific retrieval modes:

- `examples/vectorstores/weaviate/vector_search.py`
- `examples/vectorstores/weaviate/bm25_search.py`
- `examples/vectorstores/weaviate/hybrid_search.py`
