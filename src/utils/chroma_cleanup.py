import logging
import shutil
from pathlib import Path

from config.settings import CHROMA_DB_DIR

logger = logging.getLogger(__name__)


def reset_chroma_db(db_path: Path | None = None) -> None:
    db_path = db_path or CHROMA_DB_DIR
    if not db_path.exists():
        logger.info("Chroma path does not exist: %s", db_path)
        return

    if db_path.is_dir():
        shutil.rmtree(db_path)
        logger.info("Chroma directory deleted: %s", db_path)
    else:
        db_path.unlink()
        logger.info("Chroma file deleted: %s", db_path)
