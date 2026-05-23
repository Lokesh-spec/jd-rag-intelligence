import hashlib
from src.embeddings.embedding_pipeline import get_embedding_model
from src.vectorstore.chroma_store import get_vector_store
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from config.settings import (
    LLM_MODEL,
    CONSTANT_K,
    TOP_K_PER_QUERY
)
import logging 

def generate_sub_queries(original_query: str, llm: ChatOpenAI) -> list[str]:
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system", 
                "You are an AI assistant that generates multiple search query variations based on a single input query. "
                "Generate exactly 3 distinct variations, each focusing on different keywords or angles of the query. "
                "Output each query on a new line. Do not number them.",
            ),
            (
                "human",
                "{query}"
            )
        ]
    )

    formatted_messages = prompt_template.format_messages(query=original_query)

    response = llm.invoke(formatted_messages)

    queries = [line.strip() for line in response.content.strip().split("\n") if line.strip()]

    if original_query not in queries:
        queries.append(f"- {original_query}")
    
    return queries


def rag_fusion_pipeline(query:str, logger:logging) -> list[str]:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    embedding_model = get_embedding_model()

    base_retriever = get_vector_store(embedding_model).as_retriever(
        search_kwargs={
            "k" : TOP_K_PER_QUERY
        }
    )

    rrf_scores = {}
    doc_mapping = {}


    queries = generate_sub_queries(
        original_query=query,
        llm=llm
    )
    for q in queries:

        docs = base_retriever.invoke(q)

        logger.info(f"\nQuery: {q}")

        for rank, doc in enumerate(docs[:TOP_K_PER_QUERY], start=1):

            # Stable document identifier
            doc_id = hashlib.md5(
                doc.page_content.encode("utf-8")
            ).hexdigest()

            if doc_id not in doc_mapping:
                doc_mapping[doc_id] = doc

            score = 1.0 / (CONSTANT_K + rank)

            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0) + score
            )

            logger.info(
                f"Rank={rank} | "
                f"RRF Score={score:.6f} | "
                f"Company={doc.metadata.get('company_name')}"
            )
    sorted_doc_ids = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Final reranked documents
    reranked_docs = [
        doc_mapping[doc_id]
        for doc_id, _ in sorted_doc_ids
    ]

    # Display ranking
    logger.info("\n===== FINAL RRF RANKING =====")

    for i, (doc_id, score) in enumerate(sorted_doc_ids, start=1):

        doc = doc_mapping[doc_id]

        logger.info(
            f"{i}. "
            f"{doc.metadata.get('company_name')} | "
            f"{doc.metadata.get('job_title')} | "
            f"RRF Score={score:.6f}"
        )

    seen_jobs = set()
    final_docs = []

    for doc in reranked_docs:

        job_key = (
            doc.metadata.get("company_name"),
            doc.metadata.get("job_title")
        )

        if job_key in seen_jobs:
            continue

        seen_jobs.add(job_key)
        final_docs.append(doc)

    reranked_docs = final_docs
    return reranked_docs
        