# Chunking

Chunking controls how source text is split before embedding and indexing.

## Word Chunking

```python
from hayagriva.config import ChunkingConfig
from hayagriva.core.chunker import WordChunker

chunker = WordChunker(ChunkingConfig(chunk_size=100, overlap=20))
chunks, metadata = chunker.chunk(["Your document text here."])
```

## Recursive Chunking

Recursive chunking prefers larger separators first, then falls back to smaller
ones.

```python
from hayagriva.config import ChunkingConfig
from hayagriva.core.chunker import RecursiveChunker

chunker = RecursiveChunker(
    ChunkingConfig(strategy="recursive", chunk_size=300)
)
chunks, metadata = chunker.chunk([document])
```

## Semantic Chunking

Semantic chunking uses embeddings to split where adjacent sentence similarity
drops.

```python
from hayagriva.config import ChunkingConfig
from hayagriva.core.chunker import SemanticChunker
from hayagriva.core.embeddings import SentenceTransformerEmbeddings

embedder = SentenceTransformerEmbeddings()
chunker = SemanticChunker(
    ChunkingConfig(strategy="semantic", semantic_threshold=0.7),
    embedder=embedder,
)
chunks, metadata = chunker.chunk([document])
```

## Hierarchical Chunking

Hierarchical chunking creates child chunks with parent context in metadata.

```python
from hayagriva.config import ChunkingConfig
from hayagriva.core.chunker import HierarchicalChunker

chunker = HierarchicalChunker(
    ChunkingConfig(
        strategy="hierarchical",
        parent_chunk_size=1000,
        chunk_size=200,
        overlap=20,
    )
)
chunks, metadata = chunker.chunk([document])
print(metadata[0]["parent_text"])
```
