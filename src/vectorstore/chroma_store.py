from langchain_community.vectorstores import Chroma

from config.settings import (
    COLLECTION_NAME,
    CHROMA_DB_DIR
)

def get_vector_store(embedding_model):

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DB_DIR)
    )

    return vector_store