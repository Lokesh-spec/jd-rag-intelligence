"""Streamlit UI for JD Intelligence System — retrieval and LLM summarization."""

from __future__ import annotations

import logging
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.retrievers.retriever_factory import (
    EmptyIndexError,
    retrieve_matching_jds,
)

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RETRIEVER_OPTIONS = [
    "Semantic Search",
    "Semantic Search With Threshold",
    "MMR",
    "BM25",
    "Hybrid",
    "Compressed",
]

LLM_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    """
    You are a job matching assistant.

    Using ONLY the retrieved context, identify jobs that match the user's requirements.

    For EACH matching job, extract and return:

    1. Company Name
    2. Job Title
    3. Location
    4. Core Skills
    5. Posted Date / Posting Duration
    6. Applicant Count
    7. Match Reason

    Rules:
    - Extract values exactly as they appear in the context.
    - Do not write "Not specified" unless the information is truly absent from the provided context.
    - If Applicant Count, Posted Date, or Location appear anywhere in the retrieved context, include them.
    - Do not infer or hallucinate values.
    - Return results in a structured format.

    """
)
SNIPPET_MAX_CHARS = 400


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "query_history": [],
        "last_query": "",
        "last_retriever_type": RETRIEVER_OPTIONS[0],
        "last_top_k": 3,
        "last_results": [],
        "last_llm_answer": "",
        "last_analytics": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _avg_chunk_length(documents: list[Document]) -> float:
    if not documents:
        return 0.0
    return sum(len(doc.page_content) for doc in documents) / len(documents)


def _unique_sources(documents: list[Document]) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        filename = f"{source}.txt" if source != "unknown" and not str(source).endswith(".txt") else source
        if filename not in seen:
            seen.add(filename)
            sources.append(filename)
    return sources


def _build_analytics(
    query: str,
    retriever_type: str,
    top_k: int,
    documents: list[Document],
) -> dict[str, Any]:
    return {
        "query": query,
        "retriever_type": retriever_type,
        "top_k": top_k,
        "result_count": len(documents),
        "avg_chunk_length": round(_avg_chunk_length(documents), 1),
        "sources": _unique_sources(documents),
    }


def _build_context(documents: list[Document]) -> str:
    blocks: list[str] = []

    seen_jobs = set()

    for i, doc in enumerate(documents, start=1):

        source = doc.metadata.get("source", "unknown")
        description = doc.metadata.get("description", "N/A")
        company_name = doc.metadata.get("company_name", "unknown")
        job_title = doc.metadata.get("job_title", "unknown")
        posted_date = doc.metadata.get("posted_date", "unknown")
        applicants = doc.metadata.get("applicants", "unknown")
        openings = doc.metadata.get("openings", "unknown")

        # Deduplication
        job_key = (company_name, job_title)

        if job_key in seen_jobs:
            continue

        seen_jobs.add(job_key)

        blocks.append(
            f"[Chunk {i} | Source: {source} | Description: {description} | "
            f"Company Name: {company_name} | Job Title: {job_title} | "
            f"Posted Date: {posted_date} | Applicants: {applicants} | "
            f"Openings: {openings}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(blocks)


def _generate_llm_answer(query: str, documents: list[Document]) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file before using the LLM answer feature."
        )

    context = _build_context(documents)

    print(context)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User query:\n{query}\n\n"
                f"JD context:\n{context}\n\n"
                "Provide a concise, helpful answer citing relevant roles and skills "
                "from the context only."
            )
        ),
    ]
    response = llm.invoke(messages)
    return str(response.content)


def _render_sidebar() -> None:
    st.sidebar.header("Settings")

    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    st.sidebar.markdown(
        f"**OpenAI API key:** {'Configured' if api_key_set else 'Missing'}"
    )
    st.sidebar.markdown(f"**LLM model:** `{LLM_MODEL}`")

    st.sidebar.divider()
    st.sidebar.subheader("Query history")
    history: list[dict[str, Any]] = st.session_state.query_history

    if not history:
        st.sidebar.caption("No queries yet. Submit a search to build history.")
    else:
        for i, entry in enumerate(reversed(history[-10:]), start=1):
            st.sidebar.markdown(
                f"**{i}.** `{entry['retriever_type']}` — "
                f"{entry['result_count']} result(s)\n\n"
                f"_{entry['query'][:60]}{'…' if len(entry['query']) > 60 else ''}_"
            )

    st.sidebar.divider()
    clear_history = st.sidebar.button("Clear history", use_container_width=True)
    if clear_history:
        st.session_state.query_history = []
        st.rerun()

def _render_analytics_log(analytics: dict[str, Any]) -> None:
    st.subheader("Retriever analytics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Retriever", analytics["retriever_type"])
    col2.metric("Results", analytics["result_count"])
    col3.metric("Avg chunk length", f"{analytics['avg_chunk_length']:.0f} chars")
    col4.metric("Top-K requested", analytics["top_k"])

    st.markdown(f"**Query:** {analytics['query']}")
    sources = analytics["sources"]
    if sources:
        st.markdown("**Sources returned:** " + ", ".join(f"`{s}`" for s in sources))
    else:
        st.markdown("**Sources returned:** _none_")


def _render_result_cards(documents: list[Document]) -> None:
    st.subheader(f"Retrieved chunks ({len(documents)})")
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        description = doc.metadata.get("Description", "N/A")
        filename = f"{source}.txt" if not str(source).endswith(".txt") else source
        snippet = doc.page_content
        if len(snippet) > SNIPPET_MAX_CHARS:
            snippet = snippet[:SNIPPET_MAX_CHARS] + "…"

        with st.container(border=True):
            st.markdown(f"#### Result {i}")
            st.markdown(f"**Source:** `{filename}`")
            st.markdown(f"**Description:** {description}")
            st.markdown("**Content snippet:**")
            st.text(snippet)


def main() -> None:
    st.set_page_config(
        page_title="JD Intelligence System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()

    st.title("JD Intelligence System")
    st.caption(
        "Search indexed job descriptions with multiple retrievers and get an LLM-grounded summary."
    )

    _render_sidebar()

    with st.form("query_form", clear_on_submit=False):
        col_query, col_settings = st.columns([2, 1])

        with col_query:
            query = st.text_area(
                "Job-related query",
                value=st.session_state.last_query,
                height=100,
                placeholder="e.g. Looking for a GCP data engineer role without Spark or Databricks",
            )

        with col_settings:
            retriever_type = st.selectbox(
                "Retriever type",
                options=RETRIEVER_OPTIONS,
                index=RETRIEVER_OPTIONS.index(st.session_state.last_retriever_type)
                if st.session_state.last_retriever_type in RETRIEVER_OPTIONS
                else 0,
            )
            top_k = st.number_input(
                "Top-K results",
                min_value=1,
                max_value=10,
                value=int(st.session_state.last_top_k),
                step=1,
            )

        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

    if submitted:
        query = (query or "").strip()
        if not query:
            st.warning("Please enter a job-related query before searching.")
            return

        st.session_state.last_query = query
        st.session_state.last_retriever_type = retriever_type
        st.session_state.last_top_k = int(top_k)

        documents: list[Document] = []
        try:
            with st.spinner("Retrieving matching job descriptions…"):
                documents = retrieve_matching_jds(
                    query=query,
                    retriever_type=retriever_type,
                    top_k=int(top_k),
                )
        except EmptyIndexError:
            st.warning(
                "No documents found in the vector index. "
                "Run the ingestion pipeline first: `python main.py`"
            )
            return
        except EnvironmentError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            logger.exception("Retrieval failed")
            st.error(f"Retrieval failed: {exc}")
            return

        st.session_state.last_results = documents
        analytics = _build_analytics(query, retriever_type, int(top_k), documents)
        st.session_state.last_analytics = analytics
        st.session_state.query_history.append(analytics)

        if not documents:
            st.warning(
                "No results returned for this query and retriever configuration. "
                "Try a different query, retriever type, or increase Top-K."
            )
            st.session_state.last_llm_answer = ""
            return

        _render_analytics_log(analytics)
        _render_result_cards(documents)

        if not os.getenv("OPENAI_API_KEY"):
            st.info("Set OPENAI_API_KEY in `.env` to enable the LLM answer section.")
            return

        try:
            with st.spinner("Generating LLM answer from retrieved context…"):
                answer = _generate_llm_answer(query, documents)
            st.session_state.last_llm_answer = answer
        except Exception as exc:
            logger.exception("LLM answer generation failed")
            st.error(f"LLM answer failed: {exc}")
            return

        st.subheader("LLM answer")
        st.markdown(answer)

    elif st.session_state.last_results and st.session_state.last_analytics:
        st.info("Showing results from your last search. Submit a new query to refresh.")
        _render_analytics_log(st.session_state.last_analytics)
        _render_result_cards(st.session_state.last_results)
        if st.session_state.last_llm_answer:
            st.subheader("LLM answer")
            st.markdown(st.session_state.last_llm_answer)


if __name__ == "__main__":
    main()
