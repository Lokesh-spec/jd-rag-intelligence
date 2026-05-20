import logging
import os
from typing import Any

from config.settings import CLEANED_DATA_DIR, RAW_DATA_DIR
from src.loaders.document_loader import load_and_normalize_jd
from src.utils.checkpoint_manager import save_checkpoint

logger = logging.getLogger(__name__)


def run_normalize_pipeline(checkpoint: list[dict[str, Any]]) -> int:
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    state = checkpoint[0]
    ingested = set(state["ingested_raw_filenames"])
    processed = 0

    for filename in sorted(os.listdir(RAW_DATA_DIR)):
        if filename.startswith(".") or filename in ingested:
            continue

        full_path = RAW_DATA_DIR / filename
        if not full_path.is_file():
            continue

        cleaned = load_and_normalize_jd(full_path)
        output_path = CLEANED_DATA_DIR / f"{full_path.stem}.txt"
        output_path.write_text(cleaned, encoding="utf-8")

        state["ingested_raw_filenames"].append(filename)
        processed += 1
        logger.info("Normalized: %s -> %s", filename, output_path.name)

    save_checkpoint(checkpoint)
    logger.info("Normalize phase complete: %d new file(s).", processed)
    return processed
