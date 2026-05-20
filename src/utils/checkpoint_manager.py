import json
import logging
from pathlib import Path
from typing import Any

from config.settings import CHECKPOINT_PATH

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT: list[dict[str, list[str]]] = [
    {
        "ingested_raw_filenames": [],
        "normalized_output_filenames": [],
    }
]


def load_checkpoint(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or CHECKPOINT_PATH
    if not path.exists():
        logger.info("No checkpoint at %s; using default.", path)
        return [dict(DEFAULT_CHECKPOINT[0])]

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        return [dict(DEFAULT_CHECKPOINT[0])]

    entry = data[0]
    entry.setdefault("ingested_raw_filenames", [])
    entry.setdefault("normalized_output_filenames", [])
    return data


def save_checkpoint(
    checkpoint: list[dict[str, Any]],
    path: Path | None = None,
) -> None:
    path = path or CHECKPOINT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=4)
    logger.info("Checkpoint saved to %s", path)
