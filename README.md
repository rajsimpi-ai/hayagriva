# Hayagriva
Illuminate what was forgotten. Retrieve what matters.

Hayagriva is a lightweight, modular Retrieval-Augmented Generation (RAG) framework designed to help developers build production-ready AI systems that combine large language models with real-world contextual knowledge. Inspired by Hayagriva, the divine preserver and restorer of the Vedas, this library focuses on retrieving truth from stored knowledge and generating accurate, grounded responses.

## Features
- Pluggable retrievers (vector databases, keyword search, hybrid search)
- Model-agnostic generation (choose any LLM backend)
- Flexible embeddings and rerankers
- Built-in document loaders and chunking utilities
- Minimal and extensible architecture
- Easy integration with existing pipelines
- Lightweight and fast

## Quick Start
```python
from hayagriva import Hayagriva

rag = Hayagriva(
    embedding_model="all-MiniLM-L6-v2",
    llm_model="gpt-4o",
    vector_store="faiss"
)

rag.add_documents(["Hayagriva restores forgotten knowledge."])
response = rag.ask("Who retrieved the lost Vedas?")
print(response)
```

## Architecture
Documents → Chunking → Embeddings → Vector Database
                                         ↓
                                   Retriever
                                         ↓
                                 LLM Generator

## Installation
```bash
pip install hayagriva
```
(or whichever package name you publish)

## Name Origin
Hayagriva is an avatar of Vishnu, revered as the Lord of Knowledge who recovered the stolen Vedas and restored them to humanity. This library reflects the same mission:

Retrieve knowledge. Restore truth. Generate clarity.

## Contributing
Contributions, suggestions, and issue reports are welcome.

## License
MIT License

## Roadmap
- Built-in document ingestion CLI
- Streaming responses
- Web UI playground
- Auto-orchestrated multi-retriever routing
