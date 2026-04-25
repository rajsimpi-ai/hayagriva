# Vector Stores

Hayagriva currently supports FAISS and Weaviate.

## FAISS

FAISS is the default vector store and is useful for local experiments.

```python
from hayagriva.core.vectorstores import FaissVectorStore

store = FaissVectorStore()
store.add(embeddings, chunks, metadata=metadata)
results = store.search(query_embedding, top_k=4)
```

## Weaviate

Weaviate supports vector, BM25, and hybrid retrieval.

```python
from hayagriva.config import WeaviateConfig
from hayagriva.core.vectorstores import WeaviateVectorStore

store = WeaviateVectorStore(
    WeaviateConfig(
        url="http://localhost:8080",
        index_name="HayagrivaDocs",
    )
)
store.add(embeddings, chunks, metadata=metadata)
```

Start a local Weaviate service with:

```bash
bash setup/start_weaviate.sh
```

Runnable examples:

- `examples/indexing/faiss_index_chunks.py`
- `examples/indexing/weaviate_index_chunks.py`
- `examples/retrieval/faiss_retrieve_chunks.py`
- `examples/retrieval/weaviate_retrieve_chunks.py`
- `examples/vectorstores/weaviate/`
