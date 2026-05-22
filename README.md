# JD Intelligence System

RAG-style pipeline for **job description (JD) intelligence**: ingest and index JD documents with OpenAI embeddings and Chroma, then retrieve relevant roles using dense search, sparse search, hybrid strategies, and LLM-grounded summaries.

| Layer | Status | Location |
|-------|--------|----------|
| **Ingestion** | Production CLI | `main.py`, `src/` |
| **Retrieval** | Production API & UI | `src/retrievers/`, `app.py` |
| **Retrieval experiments** | Notebook evaluation | `notebooks/jd-intelligence-system.ipynb` |

## Overview

This project builds a searchable knowledge base from job posting text files. A modular Python pipeline normalizes raw JDs, chunks them with structured metadata, embeds them with `text-embedding-3-small` (500 dimensions), and persists vectors in Chroma.

**Ingestion (CLI)** — checkpointed two-phase pipeline: raw → cleaned text → chunked documents with metadata → vector index.

**Retrieval (app & library)** — query the index through a shared retriever factory used by both the Streamlit UI and your own scripts:

- **Dense retrieval** — similarity search and score-threshold filtering
- **MMR** — maximal marginal relevance for diverse results
- **Sparse retrieval** — BM25 over all indexed chunks
- **Hybrid retrieval** — weighted ensemble (Chroma + BM25, default 0.8 / 0.2)
- **Contextual compression** — LLM-based extraction of query-relevant spans

**Notebook** — step through the same ingestion and retrieval patterns interactively; useful for comparing strategies before changing defaults in `config/settings.py`.

The CLI (`main.py`) handles ingestion only. Use `app.py` or `retrieve_documents()` for search.

## Features

### Ingestion

- **Two-phase pipeline** — normalize raw JDs, then index cleaned files into Chroma
- **Checkpoint tracking** — skip already-processed files via `checkpoint.json`
- **Configurable chunking** — 500-token chunks, 50 overlap (`config/settings.py`)
- **Embeddings** — `text-embedding-3-small` at 500 dimensions
- **Chunk metadata** — each chunk carries company, role, posting signals, and a human-readable description derived from the filename

### Retrieval (production)

- Central **`RetrieverType`** enum and factory in `src/retrievers/`
- Six strategies exposed in Streamlit and via `retrieve_documents()`
- Tunable defaults for top-K, score threshold, MMR `lambda_mult`, and hybrid weights in `config/settings.py`
- **Streamlit app** — result cards, retriever analytics, query history, and GPT-4o-mini answers grounded on retrieved chunks
- **Job-level deduplication** in the LLM context builder (one entry per `company_name` + `job_title` pair)

### Retrieval (notebook)

- Mirrors ingestion and retrieval flows for experimentation
- Side-by-side comparison of dense, threshold, MMR, BM25, hybrid, and compression retrievers

## JD file naming and metadata

Place raw `.txt` files under `data/raw_jds/`. Filenames should follow:

```text
CompanyName_JobTitle.txt
```

Examples: `WissenTechnology_DataEngineer.txt`, `Accenture_CustomSoftwareEngineer.txt`.

During indexing, `src/chunking/text_splitter.py` attaches metadata to every chunk:

| Field | Source |
|-------|--------|
| `company_name` | Stem before the first `_` |
| `job_title` | Stem after the first `_` |
| `description` | Generated label, e.g. `WissenTechnology DataEngineer job description` |
| `source` | Full filename stem (used as document key) |
| `posted_date` | Parsed from `Posted:` … `Openings:` or `Applicants:` |
| `openings` | Parsed from `Openings:` … `Applicants:` (if present) |
| `applicants` | Parsed from `Applicants:` … `Save` |

Cleaned JD text should retain those `Posted:` / `Openings:` / `Applicants:` markers so metadata extraction works. After a schema change, re-run ingestion (and use `--reset-chroma` if you need a full re-index).

## Project structure

```
jd-intelligence-system/
├── config/
│   └── settings.py              # Paths, models, chunk & retrieval defaults
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
│   │   └── text_splitter.py     # chunking + JD metadata extraction
│   ├── embeddings/
│   │   └── embedding_pipeline.py
│   ├── vectorstore/
│   │   └── chroma_store.py
│   ├── indexing/
│   │   └── index_pipeline.py
│   ├── retrievers/
│   │   ├── retriever_factory.py # build_* and retrieve_documents()
│   │   ├── types.py             # RetrieverType enum
│   │   └── __init__.py
│   └── utils/
│       ├── checkpoint_manager.py
│       └── chroma_cleanup.py
├── app.py                       # Streamlit UI
├── checkpoint.json              # Runtime ledger (gitignored)
├── main.py                      # CLI ingestion entry point
└── pyproject.toml
```

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) or `pip` for dependencies
- OpenAI API key (embeddings, compression, and Streamlit LLM answers)

Key dependencies: `langchain`, `langchain-classic`, `langchain-community`, `langchain-openai`, `chromadb`, `rank-bm25`, `streamlit`.

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

   Copy `.env.example` to `.env` in the project root:

   ```env
   OPENAI_API_KEY=your_key_here
   ```

4. **Add job descriptions**

   Place `.txt` files under `data/raw_jds/` using the `CompanyName_JobTitle.txt` convention. The pipeline writes cleaned copies to `data/cleaned_jds/` and indexes them into `data/chroma_db/`.

## Usage

Run from the project root using the project interpreter:

```bash
.venv/bin/python main.py
```

### CLI options (ingestion)

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

### Streamlit app (retrieval)

After ingestion, launch the interactive UI:

```bash
.venv/bin/streamlit run app.py
```

| Retriever (UI label) | Behavior |
|----------------------|----------|
| Semantic Search | Dense similarity (`k` = Top-K) |
| Semantic Search With Threshold | Dense search with minimum relevance score |
| MMR | Diverse results via maximal marginal relevance |
| BM25 | Sparse lexical search over all chunks |
| Hybrid | Ensemble: 80% dense + 20% BM25 (configurable) |
| Compressed | Dense base + `LLMChainExtractor` compression |

The app shows retrieved chunk cards, per-query analytics (sources, avg chunk length), query history, and a structured LLM summary (company, title, location, skills, posting info, match reason) using only retrieved context.

### Programmatic retrieval

```python
from src.retrievers import retrieve_documents, RetrieverType

docs = retrieve_documents(
    query="GCP data engineer without Spark or Databricks",
    retriever_type=RetrieverType.HYBRID,  # or "Hybrid", "hybrid", etc.
    top_k=5,
)
for doc in docs:
    print(doc.metadata.get("company_name"), doc.metadata.get("job_title"))
    print(doc.page_content[:200])
```

Alias: `retrieve_matching_jds()` (used by `app.py`). Lower-level builders (`build_dense_retriever`, `build_bm25_retriever`, …) are exported from `src.retrievers` for custom pipelines.

## Configuration

Defaults live in `config/settings.py`:

| Setting | Default | Used for |
|---------|---------|----------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embeddings |
| `EMBEDDING_DIMENSIONS` | `500` | Embedding size |
| `COLLECTION_NAME` | `jd_intelligence_baseline_500d` | Chroma collection |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Text splitting |
| `DEFAULT_TOP_K` | `5` | Retrieval when top-K not specified |
| `SIMILARITY_SCORE_THRESHOLD` | `0.5` | Threshold retriever |
| `MMR_LAMBDA_MULT` | `0.5` | MMR diversity (0 = diverse, 1 = relevant) |
| `MMR_FETCH_K_MULTIPLIER` | `2` | MMR candidate pool size |
| `HYBRID_DENSE_WEIGHT` / `HYBRID_SPARSE_WEIGHT` | `0.8` / `0.2` | Hybrid ensemble |
| `LLM_TEMPERATURE` | `0` | Compression extractor LLM |

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

After running ingestion, open `notebooks/jd-intelligence-system.ipynb` to experiment with queries against the Chroma collection. Example query:

```text
Looking for a GCP data engineer role without Spark or Databricks as a skill
```

The notebook walks through normalize → index → dense similarity → threshold → MMR → BM25 → ensemble (0.8 dense / 0.2 sparse) → contextual compression. Use it to compare recall, diversity, and snippet quality; production defaults are wired through `src/retrievers/retriever_factory.py` and `config/settings.py`.

## What is gitignored

Per `.gitignore`:

- `data/` — raw, cleaned, and Chroma artifacts stay local
- `checkpoint.json` — machine-specific run state
- `.env`, `.venv/` — secrets and environment

Commit application code, config, and notebooks; keep JD content and vector DB on your machine.
