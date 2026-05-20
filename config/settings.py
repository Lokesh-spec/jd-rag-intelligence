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
