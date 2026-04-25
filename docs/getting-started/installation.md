# Installation

Hayagriva requires Python 3.10 or newer.

## Install From PyPI

```bash
pip install hayagriva
```

## Local FAISS Setup

For local examples that use sentence-transformers and FAISS:

```bash
pip install "hayagriva[cpu]"
```

For development from this repository:

```bash
bash setup/install_local_deps.sh
```

## Weaviate Setup

Weaviate examples need the local dependencies and a running Weaviate instance:

```bash
bash setup/install_local_deps.sh
bash setup/start_weaviate.sh
```

Then set the default local URL:

```bash
export WEAVIATE_URL="http://localhost:8080"
```

Stop the local Weaviate service with:

```bash
bash setup/stop_weaviate.sh
```

## Groq API Key

End-to-end generation examples require a Groq API key:

```bash
export GROQ_API_KEY="your-groq-api-key"
```

## Documentation Setup

To work on these docs locally:

```bash
pip install ".[docs]"
mkdocs serve
```

Build the static site with:

```bash
mkdocs build
```
