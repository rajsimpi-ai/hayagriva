# Configuration

Hayagriva is configured with `HayagrivaConfig` and smaller focused config
objects.

## Main Configuration

```python
from hayagriva import HayagrivaConfig

config = HayagrivaConfig(
    backend="groq",
    api_key="YOUR_GROQ_KEY",
    model="llama-3.1-8b-instant",
    vector_store="faiss",
)
```

## Chunking Configuration

```python
from hayagriva.config import ChunkingConfig

chunking = ChunkingConfig(
    strategy="recursive",
    chunk_size=500,
    overlap=50,
)
```

Supported strategies:

- `word`
- `recursive`
- `semantic`
- `hierarchical`

## Retrieval Configuration

```python
from hayagriva.config import RetrievalConfig

retrieval = RetrievalConfig(
    strategy="vector",
    top_k=4,
    similarity_threshold=0.0,
)
```

Supported strategies:

- `vector`: FAISS and Weaviate
- `bm25`: Weaviate
- `hybrid`: Weaviate

## Weaviate Configuration

```python
from hayagriva.config import WeaviateConfig

weaviate = WeaviateConfig(
    url="http://localhost:8080",
    index_name="HayagrivaDocs",
)
```

For hosted Weaviate:

```python
weaviate = WeaviateConfig(
    url="https://your-cluster.weaviate.network",
    api_key="YOUR_WEAVIATE_API_KEY",
    index_name="HayagrivaDocs",
)
```
