from pathlib import Path

from langchain_community.document_loaders import TextLoader


def load_and_normalize_jd(document_path: Path) -> str:
    loader = TextLoader(str(document_path), encoding="utf-8")
    documents = loader.load()
    text_data = documents[0].page_content

    blocks = [line for line in text_data.split("\n\n") if line.strip()]
    final_text: list[str] = []

    for block in blocks:
        block = block.strip()
        if "\n" in block:
            sub_lines = block.split("\n")
            block = " ".join(sl.strip() for sl in sub_lines if sl.strip())
        final_text.append(block)

    return " ".join(final_text)
