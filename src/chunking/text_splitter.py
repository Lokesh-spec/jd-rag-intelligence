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
    metadata = {}
    cleaned_text = source_path.read_text(encoding="utf-8")

    if "Openings:" in cleaned_text:

        metadata["posted_date"] = (
            cleaned_text.split("Posted:")[1]
            .split("Openings:")[0]
            .strip()
        )

        metadata["openings"] = (
            cleaned_text.split("Openings:")[1]
            .split("Applicants:")[0]
            .strip()
        )

    else:

        metadata["posted_date"] = (
            cleaned_text.split("Posted:")[1]
            .split("Applicants:")[0]
            .strip()
        )

        metadata["openings"] = None


    metadata["applicants"] = (
        cleaned_text.split("Applicants:")[1]
        .split("Save")[0]
        .strip()
    )
    metadata["source"] = source_path.stem
    metadata["description"] = _description_from_stem(source_path.stem)
    parts = source_path.stem.split("_", 1)

    metadata["company_name"] = parts[0]
    metadata["job_title"] = parts[1]
    return splitter.create_documents(
        texts=[cleaned_text],
        metadatas=[
            metadata
        ],
    )
