# Hayagriva

Hayagriva is a lightweight, modular Retrieval-Augmented Generation framework.
It connects document ingestion, chunking, embeddings, vector retrieval, and Groq
generation behind a small Python API.

## What Hayagriva Supports

- Groq generation backend
- FAISS local vector search
- Weaviate vector, BM25, and hybrid retrieval
- Word, recursive, semantic, and hierarchical chunking
- File and directory ingestion for `.txt` documents
- Lower-level building blocks for ingestion, chunking, indexing, and retrieval

## Quick Example

```python
from hayagriva import Hayagriva, HayagrivaConfig

rag = Hayagriva(
    HayagrivaConfig(
        backend="groq",
        api_key="YOUR_GROQ_KEY",
        model="llama-3.1-8b-instant",
    )
)

rag.add_documents(["Hayagriva restores forgotten knowledge."])
answer = rag.ask("What does Hayagriva restore?")
print(answer)
```

## Where To Go Next

- Start with [Installation](getting-started/installation.md).
- Learn the full flow in [Basic RAG](guides/basic-rag.md).
- Explore focused workflows in [Examples](examples/index.md).
- See available objects in [API Reference](reference/api.md).
