# Examples

The examples are grouped by the part of the RAG pipeline they demonstrate.

## `ingestion/`

Load raw text from Python values or `.txt` files. These examples do not call an LLM.

## `chunking/`

Split documents into chunks using Hayagriva chunkers. Basic chunking examples do not need external services; semantic chunking needs the local embedding setup.

## `indexing/`

Create embeddings and add chunks to a vector store. These examples stop before generation.

## `retrieval/`

Query an existing in-memory index and inspect retrieved chunks. These examples stop before generation.

## `end_to_end/`

Full RAG flows: ingest, chunk, embed, retrieve, and generate with Groq.

## `vectorstores/`

Provider-specific examples for external vector stores such as Weaviate.

See `setup/README.md` for dependency and service setup commands.
