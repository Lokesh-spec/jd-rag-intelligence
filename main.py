import argparse
import logging
import sys

from dotenv import load_dotenv

from src.indexing.index_pipeline import run_index_pipeline
from src.preprocessing.normalize_pipeline import run_normalize_pipeline
from src.utils.checkpoint_manager import load_checkpoint
from src.utils.chroma_cleanup import reset_chroma_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="JD Intelligence ingestion pipeline")
    parser.add_argument(
        "--reset-chroma",
        action="store_true",
        help="Delete existing Chroma persist directory before indexing",
    )
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="Run raw -> cleaned only",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Run cleaned -> Chroma only",
    )
    args = parser.parse_args()

    if args.normalize_only and args.index_only:
        logger.error("Cannot use --normalize-only and --index-only together.")
        return 1

    if args.reset_chroma:
        reset_chroma_db()

    checkpoint = load_checkpoint()

    if args.index_only:
        run_index_pipeline(checkpoint)
    elif args.normalize_only:
        run_normalize_pipeline(checkpoint)
    else:
        run_normalize_pipeline(checkpoint)
        run_index_pipeline(checkpoint)

    return 0


if __name__ == "__main__":
    sys.exit(main())
