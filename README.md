# JD Intelligence System

RAG-style pipeline for **job description (JD) intelligence**: ingest and index JD documents with OpenAI embeddings and Chroma, then retrieve relevant roles using dense search, sparse search, and hybrid strategies.

| Layer | Status | Location |
|-------|--------|----------|
| **Ingestion** | Production CLI | `main.py`, `src/` |
| **Retrieval** | Experiments & evaluation | `notebooks/jd-intelligence-system.ipynb` |

## Overview

This project builds a searchable knowledge base from job posting text files. A modular Python pipeline normalizes raw JDs, chunks them, embeds them with `text-embedding-3-small` (500 dimensions), and persists vectors in Chroma. A companion Jupyter notebook extends the same stack with retrieval patterns used in real-world RAG systems.

**Ingestion (CLI)** — checkpointed two-phase pipeline: raw → cleaned text → chunked documents → vector index.

**Retrieval (notebook)** — query the index with multiple strategies and compare behavior:

- **Dense retrieval** — similarity search with L2 scores, score-threshold filtering, and MMR (maximal marginal relevance) for diversity
- **Sparse retrieval** — BM25 over ingested chunks
- **Hybrid retrieval** — weighted ensemble (Chroma + BM25)
- **Contextual compression** — LLM-based extraction of query-relevant spans from retrieved documents

## Features

### Ingestion

- **Two-phase pipeline** — normalize raw JDs, then index cleaned files into Chroma
- **Checkpoint tracking** — skip already-processed files via `checkpoint.json`
- **Configurable chunking** — 500-token chunks, 50 overlap (`config/settings.py`)
- **Embeddings** — `text-embedding-3-small` at 500 dimensions

### Retrieval (notebook)

- Similarity search with distance scores
- Similarity with score threshold
- MMR with tunable `lambda_mult`
- BM25 sparse retriever
- Ensemble retriever (dense + sparse weights)
- Contextual compression with `LLMChainExtractor`

## Project structure

```
jd-intelligence-system/
├── config/
│   └── settings.py              # Paths, model, chunk settings
├── data/                        # Local only (gitignored)
│   ├── raw_jds/                 # Source JD text files
│   ├── cleaned_jds/             # Normalized output
│   └── chroma_db/               # Vector store persistence
├── notebooks/
│   └── jd-intelligence-system.ipynb
├── src/
│   ├── loaders/
│   │   └── document_loader.py   # load_and_normalize_jd()
│   ├── preprocessing/
│   │   └── normalize_pipeline.py
│   ├── chunking/
│   │   └── text_splitter.py
│   ├── embeddings/
│   │   └── embedding_pipeline.py
│   ├── vectorstore/
│   │   └── chroma_store.py
│   ├── indexing/
│   │   └── index_pipeline.py
│   └── utils/
│       ├── checkpoint_manager.py
│       └── chroma_cleanup.py
├── checkpoint.json              # Runtime ledger (gitignored)
├── main.py                      # CLI entry point
└── pyproject.toml
```

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) or `pip` for dependencies
- OpenAI API key

## Setup

1. **Clone and enter the project**

   ```bash
   cd jd-intelligence-system
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   uv sync
   # or: python -m venv .venv && source .venv/bin/activate && pip install -e .
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your_key_here
   ```

4. **Add job descriptions**

   Place `.txt` files under `data/raw_jds/`. The pipeline will write cleaned copies to `data/cleaned_jds/` and index them into `data/chroma_db/`.

## Usage

Run from the project root using the project interpreter:

```bash
.venv/bin/python main.py
```

### CLI options

| Flag | Description |
|------|-------------|
| *(none)* | Full pipeline: normalize, then index |
| `--normalize-only` | Phase 1 only: raw → cleaned |
| `--index-only` | Phase 2 only: cleaned → Chroma |
| `--reset-chroma` | Delete `data/chroma_db/` before indexing |

Examples:

```bash
# Full ingestion
.venv/bin/python main.py

# Re-index cleaned files only (e.g. after --reset-chroma)
.venv/bin/python main.py --reset-chroma --index-only

# Process new raw files without re-embedding existing cleaned files
.venv/bin/python main.py --normalize-only
```

## Checkpoint format

`checkpoint.json` tracks progress and is recreated on first run if missing:

```json
[
  {
    "ingested_raw_filenames": [],
    "normalized_output_filenames": []
  }
]
```

- `ingested_raw_filenames` — files processed in the normalize phase  
- `normalized_output_filenames` — cleaned files indexed into Chroma  

To reprocess everything, delete `checkpoint.json` and optionally run with `--reset-chroma`.

## Retrieval workflow (notebook)

After running ingestion, open `notebooks/jd-intelligence-system.ipynb` to experiment with queries against the Chroma collection. Example query from the notebook:

```text
Looking for a GCP data engineer Role without Spark or Databricks as a skill
```

The notebook walks through each retriever type in order: dense similarity → threshold → MMR → BM25 → ensemble (0.8 dense / 0.2 sparse) → contextual compression. Use it to compare recall, diversity, and snippet quality before promoting patterns into `src/`.

Retrieval modules are not yet exposed via `main.py`; the CLI covers ingestion only.

## What is gitignored

Per `.gitignore`:

- `data/` — raw, cleaned, and Chroma artifacts stay local  
- `checkpoint.json` — machine-specific run state  
- `.env`, `.venv/` — secrets and environment  

Commit application code, config, and notebooks; keep JD content and vector DB on your machine.
