# JD Intelligence System

RAG-style pipeline for **job description (JD) intelligence**: ingest and index JD documents with OpenAI embeddings and Chroma, then retrieve relevant roles using dense search, sparse search, hybrid strategies, RAG Fusion, and LLM-grounded summaries.

| Layer | Status | Location |
|-------|--------|----------|
| **Ingestion** | CLI | `main.py`, `src/` |
| **Retrieval** | API & UI | `src/retrievers/`, `app.py` |
| **RAG Fusion** | API & UI | `src/retrievers/rag_fusion.py` |
| **Evaluation** | RAGAS | `notebooks/ragas_evaluation.ipynb` |
| **Retrieval experiments** | Notebook | `notebooks/jd-intelligence-system.ipynb` |

## Overview

This project builds a searchable knowledge base from job posting text files. A modular Python pipeline normalizes raw JDs, chunks them with structured metadata, embeds them with `text-embedding-3-small` (500 dimensions), and persists vectors in Chroma.

**Ingestion (CLI)** — checkpointed two-phase pipeline: raw → cleaned text → chunked documents with metadata → vector index.

**Retrieval (app & library)** — query the index through a shared retriever factory used by both the Streamlit UI and your own scripts:

- **Dense retrieval** — similarity search and score-threshold filtering
- **MMR** — maximal marginal relevance for diverse results
- **Sparse retrieval** — BM25 over all indexed chunks
- **Hybrid retrieval** — weighted ensemble (Chroma + BM25, default 0.8 / 0.2)
- **Contextual compression** — LLM-based extraction of query-relevant spans
- **RAG Fusion** — multi-query generation + Reciprocal Rank Fusion (RRF) reranking

**Notebook** — step through ingestion and retrieval patterns interactively; useful for comparing strategies before changing defaults in `config/settings.py`.

The CLI (`main.py`) handles ingestion only. Use `app.py` or `retrieve_documents()` for search.

## Features

### Ingestion

- **Two-phase pipeline** — normalize raw JDs, then index cleaned files into Chroma
- **Checkpoint tracking** — skip already-processed files via `checkpoint.json`
- **Configurable chunking** — 500-token chunks, 50 overlap (`config/settings.py`)
- **Embeddings** — `text-embedding-3-small` at 500 dimensions
- **Chunk metadata** — each chunk carries company, role, posting date, applicant count, openings, and a human-readable description derived from the filename

### Retrieval

- Central **`RetrieverType`** enum and factory in `src/retrievers/`
- Seven strategies exposed in Streamlit and via `retrieve_documents()`
- Tunable defaults for top-K, score threshold, MMR `lambda_mult`, and hybrid weights in `config/settings.py`
- **Streamlit app** — result cards, retriever analytics, query history, and GPT-4o-mini answers grounded on retrieved chunks
- **Job-level deduplication** in the LLM context builder (one entry per `company_name` + `job_title` pair)

### RAG Fusion

RAG Fusion extends standard retrieval by generating multiple sub-queries from the original query, retrieving independently for each, and reranking results using **Reciprocal Rank Fusion (RRF)**.

```
Original query
      ↓
LLM generates 3 sub-query variations
      ↓
Each sub-query retrieves Top-K docs independently
      ↓
RRF scores each doc: score = 1 / (k + rank)
      ↓
Docs appearing across multiple queries ranked higher
      ↓
Final deduplicated reranked results
```

Key implementation details:
- Sub-queries generated via `ChatOpenAI` with a focused prompt
- Doc identity tracked via MD5 hash of `page_content` — prevents duplicates across queries
- Final deduplication by `(company_name, job_title)` pair — one result per role
- Configurable via `CONSTANT_K` and `TOP_K_PER_QUERY` in `config/settings.py`

### RAGAS Evaluation

Retrieval quality measured using the [RAGAS](https://docs.ragas.io) framework across a custom 20-question evaluation dataset generated from real indexed JDs.

#### Evaluation dataset

20 questions across 3 categories:
- **Simple factual** — company, role, posting date, applicant count
- **Metadata filtering** — hybrid work, immediate joiners, applicant thresholds
- **Skill-based reasoning** — multi-skill matching, comparative analysis, seniority fit

#### Results

| Metric | Hybrid Search | RAG Fusion |
|--------|--------------|------------|
| Faithfulness | 0.58 | 0.55 |
| Answer Relevancy | 0.92 | 0.92 |
| Context Precision | 0.45 | 0.45 |
| Context Recall | 0.57 | 0.57 |

**Key finding:** Hybrid Search slightly outperformed RAG Fusion on this dataset — suggesting that for short, structured documents like JDs, weighted ensemble retrieval is more effective than multi-query fusion at this corpus size.

#### Optimization experiments & learnings

Three optimization approaches were tested to improve scores:

**1. Increasing top_k (3 → 5)**
Hypothesis: more chunks retrieved = better recall.
Result: Context Precision dropped from 0.45 → 0.37. More chunks introduced irrelevant JDs into context, adding noise for the LLM.
Learning: `top_k=3` is optimal for a 25-JD corpus. Precision/recall tradeoff shifts with larger datasets.

**2. Refining RAG Fusion sub-query prompt**
Hypothesis: more specific sub-queries = more precise retrieval.
Result: Answer Relevancy dropped from 0.92 → 0.88. Over-constraining sub-queries narrowed retrieval too aggressively, missing relevant chunks.
Learning: Sub-query specificity must be balanced — too broad loses precision, too narrow loses recall.

**3. Chunk size experimentation**
Hypothesis: smaller chunks = more precise retrieval.
Result: Precision improved marginally but recall dropped. 500-char chunks were optimal for JD-length documents.
Learning: Chunk size is domain-dependent. JDs are already short and structured — aggressive chunking fragments meaningful skill and responsibility blocks.

**Root cause of score ceiling:** All three experiments confirmed that retrieval optimization is dataset-size dependent. With 25 JDs, the ceiling is constrained by corpus size. A larger dataset would shift all three tradeoffs significantly.

## JD file naming and metadata

Place raw `.txt` files under `data/raw_jds/`. Filenames should follow:

```
CompanyName_JobTitle.txt
```

Examples: `WissenTechnology_DataEngineer.txt`, `Accenture_CustomSoftwareEngineer.txt`.

During indexing, `src/chunking/text_splitter.py` attaches metadata to every chunk:

| Field | Source |
|-------|--------|
| `company_name` | Stem before the first `_` |
| `job_title` | Stem after the first `_` |
| `description` | Generated label, e.g. `WissenTechnology DataEngineer job description` |
| `source` | Full filename stem |
| `posted_date` | Parsed from `Posted:` marker |
| `openings` | Parsed from `Openings:` marker |
| `applicants` | Parsed from `Applicants:` marker |

## Project structure

```
jd-intelligence-system/
├── config/
│   └── settings.py                  # Paths, models, chunk & retrieval defaults
├── data/                            # Local only (gitignored)
│   ├── raw_jds/                     # Source JD text files
│   ├── cleaned_jds/                 # Normalized output
│   └── chroma_db/                   # Vector store persistence
├── notebooks/
│   ├── jd-intelligence-system.ipynb # Retrieval experiments
│   └── ragas_evaluation.ipynb       # RAGAS evaluation pipeline
├── src/
│   ├── loaders/
│   │   └── document_loader.py       # load_and_normalize_jd()
│   ├── preprocessing/
│   │   └── normalize_pipeline.py
│   ├── chunking/
│   │   └── text_splitter.py         # Chunking + JD metadata extraction
│   ├── embeddings/
│   │   └── embedding_pipeline.py
│   ├── vectorstore/
│   │   └── chroma_store.py
│   ├── indexing/
│   │   └── index_pipeline.py
│   ├── retrievers/
│   │   ├── retriever_factory.py     # build_* and retrieve_documents()
│   │   ├── rag_fusion.py            # RAG Fusion + RRF pipeline
│   │   ├── types.py                 # RetrieverType enum
│   │   └── __init__.py
│   └── utils/
│       ├── checkpoint_manager.py
│       └── chroma_cleanup.py
├── app.py                           # Streamlit UI
├── checkpoint.json                  # Runtime ledger (gitignored)
├── main.py                          # CLI ingestion entry point
└── pyproject.toml
```

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) or `pip` for dependencies
- OpenAI API key (embeddings, compression, RAG Fusion sub-queries, and Streamlit LLM answers)

Key dependencies: `langchain`, `langchain-community`, `langchain-openai`, `chromadb`, `rank-bm25`, `ragas`, `streamlit`.

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

### Ingestion (CLI)

```bash
.venv/bin/python main.py
```

| Flag | Description |
|------|-------------|
| *(none)* | Full pipeline: normalize, then index |
| `--normalize-only` | Phase 1 only: raw → cleaned |
| `--index-only` | Phase 2 only: cleaned → Chroma |
| `--reset-chroma` | Delete `data/chroma_db/` before indexing |

### Streamlit app

```bash
.venv/bin/streamlit run app.py
```

| Retriever | Behavior |
|-----------|----------|
| Semantic Search | Dense similarity |
| Semantic Search With Threshold | Dense with minimum relevance score |
| MMR | Diverse results via maximal marginal relevance |
| BM25 | Sparse lexical search |
| Hybrid | 80% dense + 20% BM25 |
| Compressed | Dense + LLM compression |
| RAG Fusion | Multi-query + RRF reranking |

### Programmatic retrieval

```python
from src.retrievers import retrieve_documents, RetrieverType

docs = retrieve_documents(
    query="GCP data engineer without Spark or Databricks",
    retriever_type=RetrieverType.HYBRID,
    top_k=5,
)
for doc in docs:
    print(doc.metadata.get("company_name"), doc.metadata.get("job_title"))
    print(doc.page_content[:200])
```

## Configuration

| Setting | Default | Used for |
|---------|---------|----------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embeddings |
| `EMBEDDING_DIMENSIONS` | `500` | Embedding size |
| `COLLECTION_NAME` | `jd_intelligence_baseline_500d` | Chroma collection |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Text splitting |
| `DEFAULT_TOP_K` | `3` | Retrieval default |
| `SIMILARITY_SCORE_THRESHOLD` | `0.5` | Threshold retriever |
| `MMR_LAMBDA_MULT` | `0.5` | MMR diversity |
| `HYBRID_DENSE_WEIGHT` / `HYBRID_SPARSE_WEIGHT` | `0.8` / `0.2` | Hybrid ensemble |
| `CONSTANT_K` | `60` | RRF ranking constant |
| `TOP_K_PER_QUERY` | `3` | Docs per sub-query in RAG Fusion |
| `LLM_TEMPERATURE` | `0` | Compression + RAG Fusion LLM |

## Checkpoint format

```json
[
  {
    "ingested_raw_filenames": [],
    "normalized_output_filenames": []
  }
]
```

To reprocess everything, delete `checkpoint.json` and optionally run with `--reset-chroma`.

## What is gitignored

- `data/` — raw, cleaned, and Chroma artifacts stay local
- `checkpoint.json` — machine-specific run state
- `.env`, `.venv/` — secrets and environment