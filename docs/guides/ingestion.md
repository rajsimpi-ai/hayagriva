# Ingestion

Hayagriva can ingest raw strings or load `.txt` files from paths.

## Load Raw Text

```python
from hayagriva.ingestion.loaders import load_texts

documents = load_texts([
    "Hayagriva restores knowledge.",
    "RAG starts with clean source text.",
])
```

## Load Files

```python
from hayagriva.ingestion.loaders import load_from_paths

documents = load_from_paths(["./docs"])
```

Directory paths are searched recursively for `.txt` files.

Runnable examples:

- `examples/ingestion/load_texts.py`
- `examples/ingestion/load_from_files.py`
