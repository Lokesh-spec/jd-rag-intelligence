from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_jds"
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned_jds"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoint.json"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 500
COLLECTION_NAME = "jd_intelligence_baseline_500d"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval defaults (aligned with notebooks/jd-intelligence-system.ipynb)
DEFAULT_TOP_K = 5
SIMILARITY_SCORE_THRESHOLD = 0.5
MMR_LAMBDA_MULT = 0.5
MMR_FETCH_K_MULTIPLIER = 2
HYBRID_DENSE_WEIGHT = 0.8
HYBRID_SPARSE_WEIGHT = 0.2
LLM_TEMPERATURE = 0
LLM_MODEL = "gpt-4o-mini"

CONSTANT_K = 60
TOP_K_PER_QUERY = 5