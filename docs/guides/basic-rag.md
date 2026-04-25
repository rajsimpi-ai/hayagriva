# Basic RAG

This guide shows the full Hayagriva flow:

1. Configure the backend.
2. Add documents.
3. Retrieve relevant chunks.
4. Generate an answer with Groq.

```python
from hayagriva import Hayagriva, HayagrivaConfig

config = HayagrivaConfig(
    backend="groq",
    api_key="YOUR_GROQ_KEY",
    model="llama-3.1-8b-instant",
)

rag = Hayagriva(config)
rag.add_documents(
    [
        "Hayagriva is associated with restored knowledge.",
        "RAG systems retrieve context before generating an answer.",
        "FAISS is useful for local vector search.",
    ]
)

answer = rag.ask("Why does RAG retrieve context first?")
print(answer)
```

## Return Metadata

Use `return_metadata=True` to inspect retrieval and model details:

```python
response = rag.ask(
    "Why does RAG retrieve context first?",
    return_metadata=True,
)

print(response["answer"])
print(response["chunks"])
print(response["retrieval"])
```

See `examples/end_to_end/` for complete runnable scripts.
