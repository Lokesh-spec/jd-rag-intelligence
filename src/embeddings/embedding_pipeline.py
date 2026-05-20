from langchain_openai import OpenAIEmbeddings

from config.settings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL


def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        max_retries=3,
    )
