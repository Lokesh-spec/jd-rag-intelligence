import logging
import os
from typing import Any

from config.settings import CLEANED_DATA_DIR
from src.chunking.text_splitter import chunk_jd_text, get_text_splitter
from src.embeddings.embedding_pipeline import get_embedding_model
from src.utils.checkpoint_manager import save_checkpoint
from src.vectorstore.chroma_store import get_vector_store

logger = logging.getLogger(__name__)


def run_index_pipeline(checkpoint: list[dict[str, Any]]) -> int:
    state = checkpoint[0]
    indexed = set(state["normalized_output_filenames"])
    splitter = get_text_splitter()
    embeddings = get_embedding_model()
    vector_store = get_vector_store(embeddings)
    processed = 0

    for filename in sorted(os.listdir(CLEANED_DATA_DIR)):
        if filename.startswith(".") or not filename.endswith(".txt"):
            continue
        if filename in indexed:
            continue

        source_path = CLEANED_DATA_DIR / filename
        chunks = chunk_jd_text(splitter, source_path)
        if not chunks:
            logger.warning("No chunks for %s; skipping.", filename)
            continue

        chunk_ids = vector_store.add_documents(chunks)
        state["normalized_output_filenames"].append(filename)
        processed += 1
        logger.info(
            "Indexed %s: %d chunk(s), ids=%s..%s",
            filename,
            len(chunks),
            chunk_ids[0],
            chunk_ids[-1],
        )

    save_checkpoint(checkpoint)
    logger.info("Index phase complete: %d new file(s).", processed)
    return processed
