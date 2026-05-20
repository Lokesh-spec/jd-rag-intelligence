from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


def _description_from_stem(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]} job description"
    return f"{stem} job description"


def chunk_jd_text(
    splitter: RecursiveCharacterTextSplitter,
    source_path: Path,
) -> list[Document]:
    cleaned_text = source_path.read_text(encoding="utf-8")
    stem = source_path.stem
    return splitter.create_documents(
        texts=[cleaned_text],
        metadatas=[
            {
                "source": stem,
                "Description": _description_from_stem(stem),
            }
        ],
    )
